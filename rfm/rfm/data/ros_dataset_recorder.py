import json
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64


def _img_to_np(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        arr = arr.reshape((h, w, 3))
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return arr
    if msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape((h, w))
        return arr[:, :, None]
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _pose_to_pos_quat(msg: PoseStamped) -> np.ndarray:
    p = msg.pose.position
    q = msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)


class _KeyReader:
    """Non-blocking single-key reader for terminal (Linux)."""

    def __init__(self):
        self._stop = threading.Event()
        self.last_key: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while not self._stop.is_set():
                    ch = sys.stdin.read(1)
                    if ch:
                        self.last_key = ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # Fallback: no key support
            return


@dataclass
class Args:
    camera_map_file: str = "/home/lcw/RFM_lerobot/camera_port_map.json"
    topic_prefix: str = "/rs"
    robot_joint_topic: str = "/ur5/joint_state"
    robot_tcp_topic: str = "/ur5/tcp_pose"
    teleop_joint_cmd_topic: str = "/ur5/goal_joint"
    teleop_gripper_cmd_topic: str = "/ur5/gripper_cmd"
    out_dir: str = "/home/lcw/RFM_lerobot/data/ros"
    hz: float = 30.0
    save_depth: bool = True


class RosDatasetRecorder(Node):
    def __init__(self, args: Args):
        super().__init__("rfm_ros_dataset_recorder")
        self.args = args

        # Load camera names
        with open(args.camera_map_file, "r") as f:
            mapping = json.load(f)
        self.camera_names = [str(ent["name"]) for ent in mapping]
        self.get_logger().info(f"Cameras: {self.camera_names} (file={args.camera_map_file})")

        self._latest_joint: Optional[JointState] = None
        self._latest_tcp: Optional[PoseStamped] = None
        self._latest_rgb: Dict[str, Image] = {}
        self._latest_depth: Dict[str, Image] = {}
        self._latest_goal_joint: Optional[JointState] = None
        self._latest_gripper_cmd: Optional[Float64] = None

        self.create_subscription(JointState, args.robot_joint_topic, self._on_joint, 10)
        self.create_subscription(PoseStamped, args.robot_tcp_topic, self._on_tcp, 10)
        # Optional: teleop commands. If a teleop node isn't running, these may stay None.
        self.create_subscription(JointState, args.teleop_joint_cmd_topic, self._on_goal_joint, 10)
        self.create_subscription(Float64, args.teleop_gripper_cmd_topic, self._on_gripper_cmd, 10)

        for name in self.camera_names:
            self.create_subscription(
                Image,
                f"{args.topic_prefix.rstrip('/')}/{name}/color/image_raw",
                lambda msg, n=name: self._on_rgb(n, msg),
                10,
            )
            if args.save_depth:
                self.create_subscription(
                    Image,
                    f"{args.topic_prefix.rstrip('/')}/{name}/depth/image_raw",
                    lambda msg, n=name: self._on_depth(n, msg),
                    10,
                )

        # recording control
        self._recording = False
        self._quit = False
        self._keys = _KeyReader()
        self._keys.start()

        # output folder per run
        ts = time.strftime("%m%d_%H%M%S")
        self.run_dir = Path(os.path.expanduser(args.out_dir)) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._episode_idx = 0
        self._episode_dir: Optional[Path] = None

        self.get_logger().info(f"Saving to: {self.run_dir}")
        self.get_logger().info("Keys: s=start, q=stop, ESC=quit")

        self._timer = self.create_timer(1.0 / float(args.hz), self._tick)

    def _on_joint(self, msg: JointState):
        self._latest_joint = msg

    def _on_tcp(self, msg: PoseStamped):
        self._latest_tcp = msg

    def _on_goal_joint(self, msg: JointState):
        self._latest_goal_joint = msg

    def _on_gripper_cmd(self, msg: Float64):
        self._latest_gripper_cmd = msg

    def _on_rgb(self, name: str, msg: Image):
        self._latest_rgb[name] = msg

    def _on_depth(self, name: str, msg: Image):
        self._latest_depth[name] = msg

    def _tick(self):
        # handle keys
        k = self._keys.last_key
        if k is not None:
            self._keys.last_key = None
            if k.lower() == "s":
                if not self._recording:
                    ep_name = f"episode_{self._episode_idx:03d}"
                    self._episode_dir = self.run_dir / ep_name
                    self._episode_dir.mkdir(parents=True, exist_ok=True)
                    self._episode_idx += 1
                    self._recording = True
                    self.get_logger().info(f"RECORDING: ON ({self._episode_dir})")
            elif k.lower() == "q":
                if self._recording:
                    self._recording = False
                    self.get_logger().info("RECORDING: OFF")
                    self._episode_dir = None
            elif ord(k) == 27:  # ESC
                self._quit = True

        if self._quit:
            raise KeyboardInterrupt

        if not self._recording:
            return
        if self._episode_dir is None:
            return

        if self._latest_joint is None or self._latest_tcp is None:
            return

        # Require at least one frame per camera (rgb, and depth if enabled)
        for name in self.camera_names:
            if name not in self._latest_rgb:
                return
            if self.args.save_depth and name not in self._latest_depth:
                return

        # Build obs dict (same key style as ZMQ pipeline)
        out: Dict[str, object] = {}
        for name in self.camera_names:
            rgb = _img_to_np(self._latest_rgb[name])
            out[f"{name}_rgb"] = rgb
            if self.args.save_depth:
                depth = _img_to_np(self._latest_depth[name])
                out[f"{name}_depth"] = depth

        # joints (6-DoF from bridge) + add gripper placeholder
        q = np.array(list(self._latest_joint.position), dtype=np.float64)
        if q.shape[0] == 6:
            q = np.concatenate([q, np.array([0.0])], axis=0)
        out["joint_positions"] = q
        # velocity may be empty
        v = np.array(list(self._latest_joint.velocity), dtype=np.float64) if self._latest_joint.velocity else np.zeros_like(q)
        if v.shape[0] == 6:
            v = np.concatenate([v, np.array([0.0])], axis=0)
        out["joint_velocities"] = v
        out["ee_pos_quat"] = _pose_to_pos_quat(self._latest_tcp)
        out["gripper_position"] = np.array([0.0], dtype=np.float64)
        # Teleop command (if available). We store absolute target joint positions + last gripper command.
        # If teleop isn't publishing yet, fall back to zeros to keep shape stable.
        ctrl = np.zeros((7,), dtype=np.float64)
        if self._latest_goal_joint is not None and self._latest_goal_joint.position:
            cmd_q = np.array(list(self._latest_goal_joint.position), dtype=np.float64).reshape(-1)
            if cmd_q.shape[0] >= 6:
                ctrl[:6] = cmd_q[:6]
        if self._latest_gripper_cmd is not None:
            ctrl[6] = float(self._latest_gripper_cmd.data)
        out["control"] = ctrl

        stamp = self.get_clock().now().to_msg()
        fname = f"{stamp.sec}.{stamp.nanosec:09d}.pkl"
        path = self._episode_dir / fname
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--camera-map-file", default=Args.camera_map_file)
    p.add_argument("--topic-prefix", default=Args.topic_prefix)
    p.add_argument("--robot-joint-topic", default=Args.robot_joint_topic)
    p.add_argument("--robot-tcp-topic", default=Args.robot_tcp_topic)
    p.add_argument("--teleop-joint-cmd-topic", default=Args.teleop_joint_cmd_topic)
    p.add_argument("--teleop-gripper-cmd-topic", default=Args.teleop_gripper_cmd_topic)
    p.add_argument("--out-dir", default=Args.out_dir)
    p.add_argument("--hz", type=float, default=Args.hz)
    p.add_argument("--no-depth", action="store_true", default=False)
    args_ns = p.parse_args()

    args = Args(
        camera_map_file=str(args_ns.camera_map_file),
        topic_prefix=str(args_ns.topic_prefix),
        robot_joint_topic=str(args_ns.robot_joint_topic),
        robot_tcp_topic=str(args_ns.robot_tcp_topic),
        teleop_joint_cmd_topic=str(args_ns.teleop_joint_cmd_topic),
        teleop_gripper_cmd_topic=str(args_ns.teleop_gripper_cmd_topic),
        out_dir=str(args_ns.out_dir),
        hz=float(args_ns.hz),
        save_depth=(not bool(args_ns.no_depth)),
    )

    rclpy.init()
    node = RosDatasetRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

