#!/usr/bin/env python3
"""
GELLO -> UR5 (via ROS2 bridge) teleoperation adapter.

This node reads the GELLO leader device (Dynamixel) and publishes UR5 joint
targets to the ROS2 topics consumed by `ur5.robots.ur5_bridge`:

  - /ur5/goal_joint   (sensor_msgs/JointState)  absolute joint targets (rad)
  - /ur5/gripper_cmd  (std_msgs/Float64)        simple gripper command

Important:
  - Start `python -m ur5.robots.ur5_bridge` first.
  - This node does NOT connect to the robot via RTDEControl (so it won't fight
    the bridge). It only opens a read-only RTDEReceive connection to get the
    current joint state for rate limiting and auto-calibration.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String


def parse_args():
    p = argparse.ArgumentParser(description="GELLO -> ROS2 teleop adapter for UR5 bridge.")
    p.add_argument(
        "--robot-ip",
        type=str,
        required=True,
        help="UR robot IP (used only for rtde_receive state readback).",
    )
    p.add_argument(
        "--use-ros-joint-state",
        action="store_true",
        help="Use /ur5/joint_state from the bridge instead of opening a direct RTDEReceive connection. "
             "Recommended when RTDE connections are unstable or limited.",
    )
    p.add_argument(
        "--ros-joint-state-timeout",
        type=float,
        default=15.0,
        help="When --use-ros-joint-state is set, how long to wait for the first /ur5/joint_state (seconds).",
    )
    p.add_argument(
        "--ros-joint-state-stale",
        type=float,
        default=2.0,
        help="When --use-ros-joint-state is set, consider /ur5/joint_state stale after this many seconds.",
    )
    p.add_argument(
        "--gello-port",
        type=str,
        default=None,
        help="GELLO serial port (/dev/serial/by-id/...). If omitted, auto-picks the first /dev/serial/by-id/*.",
    )
    p.add_argument(
        "--calib-file",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "gello_calibration.json")),
        help="Path to saved GELLO calibration JSON (joint_offsets per port).",
    )
    p.add_argument(
        "--save-calib",
        action="store_true",
        help="Save a fixed calibration to --calib-file using current robot_q (expected: robot at observe) "
             "and current GELLO physical pose (expected: leader at home). Exits after saving.",
    )
    p.add_argument(
        "--load-calib",
        action="store_true",
        help="Load fixed calibration from --calib-file and disable auto-calibration.",
    )
    p.add_argument("--hz", type=float, default=100.0, help="Control loop frequency (Hz).")
    p.add_argument(
        "--joint-topic",
        type=str,
        default="/ur5/goal_joint",
        help="Joint command topic. Default is /ur5/goal_joint (supported by ur5.robots.ur5_bridge). "
             "If your bridge supports streaming servo control, you can set this to /ur5/servo_joint.",
    )
    p.add_argument(
        "--max-start-joint-delta",
        type=float,
        default=0.8,
        help="Startup safety check: if any joint delta between leader and robot exceeds this (rad), abort.",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=25,
        help="Number of warmup cycles before entering the steady control loop.",
    )
    p.add_argument(
        "--warmup-max-delta",
        type=float,
        default=0.05,
        help="Per-cycle max joint delta during warmup (rad).",
    )
    p.add_argument(
        "--max-joint-step",
        type=float,
        default=0.05,
        help="Per-cycle max joint delta (rad). Helps keep motion smooth/safe.",
    )
    p.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Leader gripper threshold. >threshold=open, else close.",
    )
    p.add_argument(
        "--gripper-open",
        type=float,
        default=1.0,
        help="Value published to /ur5/gripper_cmd for open.",
    )
    p.add_argument(
        "--gripper-close",
        type=float,
        default=-1.0,
        help="Value published to /ur5/gripper_cmd for close.",
    )
    p.add_argument(
        "--go-observe-on-start",
        action="store_true",
        help="Send '/ur5/cmd: go observe' once before calibration/teleop starts.",
    )
    p.add_argument(
        "--observe-wait-s",
        type=float,
        default=4.0,
        help="Seconds to wait after sending 'go observe' before starting teleop.",
    )
    return p.parse_args()


class GelloRosTeleop(Node):
    def __init__(
        self,
        robot_ip: str,
        gello_port: Optional[str],
        hz: float,
        joint_topic: str,
        max_start_joint_delta: float,
        warmup_steps: int,
        warmup_max_delta: float,
        max_joint_step: float,
        gripper_threshold: float,
        gripper_open: float,
        gripper_close: float,
        ros_joint_state_timeout: float = 15.0,
        ros_joint_state_stale: float = 2.0,
        calib_file: Optional[str] = None,
        save_calib: bool = False,
        load_calib: bool = False,
        go_observe_on_start: bool = False,
        observe_wait_s: float = 4.0,
    ):
        super().__init__("gello_ros_teleop")

        self.robot_ip = robot_ip
        self.gello_port = gello_port
        self.hz = float(hz)
        self.dt = 1.0 / max(1e-6, self.hz)
        self.joint_topic = str(joint_topic)
        self.max_start_joint_delta = float(max_start_joint_delta)
        self.warmup_steps = int(warmup_steps)
        self.warmup_max_delta = float(warmup_max_delta)
        self.max_joint_step = float(max_joint_step)
        self.gripper_threshold = float(gripper_threshold)
        self.gripper_open = float(gripper_open)
        self.gripper_close = float(gripper_close)

        # ROS pubs (consumed by ur5_bridge)
        self.pub_goal_joint = self.create_publisher(JointState, self.joint_topic, 10)
        self.pub_gripper = self.create_publisher(Float64, "/ur5/gripper_cmd", 10)

        # Robot state source:
        # - Preferred: ROS joint state published by the bridge (/ur5/joint_state)
        # - Fallback: direct read-only RTDE receive
        self.use_ros_joint_state = False
        self._latest_robot_q: Optional[np.ndarray] = None
        self._latest_robot_q_ts: float = 0.0
        self.ros_joint_state_timeout = float(ros_joint_state_timeout)
        self.ros_joint_state_stale = float(ros_joint_state_stale)
        self.calib_file = str(calib_file) if calib_file else str(Path(__file__).resolve().parents[2] / "gello_calibration.json")
        self.save_calib = bool(save_calib)
        self.load_calib = bool(load_calib)
        self.go_observe_on_start = bool(go_observe_on_start)
        self.observe_wait_s = float(observe_wait_s)

        # NOTE: args is parsed in main(); we can't see it here directly, so we
        # detect via an attribute injected by main().
        if getattr(self, "_use_ros_joint_state_flag", False):
            self.use_ros_joint_state = True

        if self.use_ros_joint_state:
            self.sub_joint_state = self.create_subscription(
                JointState, "/ur5/joint_state", self._on_robot_joint_state, 10
            )
            self.get_logger().info("Using /ur5/joint_state (ROS) for robot q (no RTDEReceive opened).")
            self.rtde_r = None
            self._wait_for_first_joint_state(timeout_s=self.ros_joint_state_timeout)
        else:
            import rtde_receive
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.get_logger().info(f"Connected RTDE receive: {self.robot_ip}")

        self.pub_cmd = self.create_publisher(String, "/ur5/cmd", 10)
        if self.go_observe_on_start:
            self._go_observe_before_start()

        # Make sure local gello_software is importable when running from repo root.
        gello_path = Path(__file__).resolve().parents[2] / "gello_software"
        if gello_path.exists():
            sys.path.insert(0, str(gello_path))

        # Instantiate GELLO agent (Dynamixel leader).
        # Calibration modes:
        # - default: auto-calibrate GELLO offsets so the current leader reading maps to current robot_q
        # - --load-calib: load saved offsets (fixed calibration), do NOT auto-calibrate
        # - --save-calib: perform auto-calib once, save offsets, then exit
        from gello.agents.gello_agent import GelloAgent

        q0 = self._get_robot_q()
        start_joints: Optional[np.ndarray]
        if self.load_calib:
            start_joints = None
            self.get_logger().info(f"Loading fixed GELLO calibration from {self.calib_file} (auto-calibration disabled).")
        else:
            start_joints = np.concatenate([q0, np.array([0.0], dtype=np.float64)])  # + gripper
            self.get_logger().info(
                "Auto-calibrating GELLO to current robot pose (start_joints = robot_q + [0.0])"
            )
        gello_port = self.gello_port
        if not gello_port:
            usb_ports = sorted(glob.glob("/dev/serial/by-id/*"))
            self.get_logger().info(f"Found {len(usb_ports)} serial port(s): {usb_ports}")
            if len(usb_ports) == 0:
                raise RuntimeError("No GELLO port found. Provide --gello-port or plug in the device.")
            gello_port = usb_ports[0]
            self.get_logger().info(f"Using GELLO serial port: {gello_port}")
            if len(usb_ports) > 1:
                self.get_logger().warn(
                    "Multiple serial ports detected — specify --gello-port /dev/serial/by-id/... if this is wrong."
                )
        self.gello_port = str(gello_port)

        self.agent = GelloAgent(port=self.gello_port, start_joints=start_joints)

        # Apply / save fixed calibration (joint offsets) if requested.
        # NOTE: GelloAgent stores the robot as `_robot`; we intentionally keep this contained here.
        leader_robot = getattr(self.agent, "_robot", None)
        if leader_robot is None:
            raise RuntimeError("Unexpected: GelloAgent has no _robot attribute")

        if self.load_calib:
            self._apply_saved_calibration(leader_robot)

        if self.save_calib:
            self._save_current_calibration(leader_robot)
            self.get_logger().info("Saved calibration; exiting as requested by --save-calib.")
            raise SystemExit(0)

        # Startup safety check: ensure leader and robot are roughly aligned after calibration.
        leader0 = np.asarray(self.agent.act({}), dtype=np.float64).reshape(-1)
        if leader0.shape[0] < 6:
            raise RuntimeError(f"Leader output dim too small at startup: {leader0.shape}")
        q_leader0 = leader0[:6]
        dq0 = q_leader0 - q0
        if float(np.max(np.abs(dq0))) > self.max_start_joint_delta:
            self.get_logger().error(
                f"Startup check failed: max(|leader-robot|)={float(np.max(np.abs(dq0))):.3f}rad "
                f"> {self.max_start_joint_delta:.3f}rad. "
                "GELLO pose may be miscalibrated; restart with GELLO aligned or replug device."
            )
            raise RuntimeError("Leader/robot mismatch too large at startup.")

        # Warmup: gently pull the robot toward the leader before starting the steady timer loop.
        for _ in range(max(0, self.warmup_steps)):
            # During __init__ we are not spinning yet, so we must manually pump callbacks
            # to keep /ur5/joint_state fresh when using ROS readback.
            if self.use_ros_joint_state:
                rclpy.spin_once(self, timeout_sec=0.0)
            q = self._get_robot_q()
            leader = np.asarray(self.agent.act({}), dtype=np.float64).reshape(-1)
            if leader.shape[0] < 6:
                continue
            q_target = leader[:6]
            dq = q_target - q
            max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
            if max_abs > self.warmup_max_delta:
                dq = dq / max_abs * self.warmup_max_delta
            q_cmd = q + dq
            msg = JointState()
            msg.position = [float(v) for v in q_cmd.tolist()]
            self.pub_goal_joint.publish(msg)
            time.sleep(self.dt)

        # Main loop timer
        self.create_timer(self.dt, self._tick)

        self._last_gripper_cmd: Optional[float] = None
        self.get_logger().info(
            f"Publishing to {self.joint_topic} @ {self.hz:.1f}Hz (max_joint_step={self.max_joint_step})"
        )

    def _go_observe_before_start(self):
        self.get_logger().info("Sending '/ur5/cmd: go observe' before teleop start.")
        msg = String()
        msg.data = "go observe"
        self.pub_cmd.publish(msg)
        wait_s = max(0.0, self.observe_wait_s)
        if wait_s <= 0.0:
            return
        t0 = time.time()
        while (time.time() - t0) < wait_s:
            # Keep callbacks flowing so /ur5/joint_state stays fresh.
            if self.use_ros_joint_state:
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)

    def _wait_for_first_joint_state(self, timeout_s: float = 5.0):
        """Block briefly until we have an initial robot joint state."""
        deadline = time.time() + float(timeout_s)
        while self._latest_robot_q is None and time.time() < deadline:
            # Allow ROS callbacks to be processed during init.
            rclpy.spin_once(self, timeout_sec=0.05)
        if self._latest_robot_q is None:
            raise RuntimeError(
                f"Timed out waiting for /ur5/joint_state ({timeout_s:.1f}s). "
                "Is `ur5.robots.ur5_bridge` running and publishing?"
            )

    def _get_robot_q(self) -> np.ndarray:
        if self.use_ros_joint_state:
            if self._latest_robot_q is None or (time.time() - self._latest_robot_q_ts) > self.ros_joint_state_stale:
                raise RuntimeError("No recent /ur5/joint_state received (is ur5_bridge running?)")
            q = self._latest_robot_q
        else:
            if self.rtde_r is None:
                raise RuntimeError("RTDEReceive not initialized")
            q = np.array(self.rtde_r.getActualQ(), dtype=np.float64)
        if q.shape != (6,):
            raise RuntimeError(f"Unexpected RTDE q shape: {q.shape}")
        return q

    def _on_robot_joint_state(self, msg: JointState):
        try:
            if not msg.position or len(msg.position) < 6:
                return
            q = np.array([float(v) for v in msg.position[:6]], dtype=np.float64)
            if q.shape != (6,):
                return
            self._latest_robot_q = q
            self._latest_robot_q_ts = time.time()
        except Exception:
            return

    def _tick(self):
        try:
            q = self._get_robot_q()

            # Leader target from GELLO (expects 7 dims: 6 joints + gripper)
            leader = self.agent.act({})  # obs is unused
            leader = np.asarray(leader, dtype=np.float64).reshape(-1)
            if leader.shape[0] < 6:
                self.get_logger().warn(f"Leader dim too small: {leader.shape}")
                return

            q_target = leader[:6]
            dq = q_target - q
            max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
            if max_abs > self.max_joint_step:
                dq = dq / max_abs * self.max_joint_step
            q_cmd = q + dq

            msg = JointState()
            msg.position = [float(v) for v in q_cmd.tolist()]
            self.pub_goal_joint.publish(msg)

            # Gripper: publish binary open/close by threshold
            if leader.shape[0] >= 7:
                g = float(leader[6])
                g_cmd = self.gripper_open if g > self.gripper_threshold else self.gripper_close
                if self._last_gripper_cmd != g_cmd:
                    gm = Float64()
                    gm.data = float(g_cmd)
                    self.pub_gripper.publish(gm)
                    self._last_gripper_cmd = g_cmd
        except Exception as e:
            self.get_logger().warn(f"tick failed: {e}")

    # ======================================================================
    # Fixed calibration (save/load)
    # ======================================================================

    def _read_calib_db(self) -> dict:
        path = Path(self.calib_file)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed to read calib file {path}: {e}")

    def _write_calib_db(self, data: dict) -> None:
        path = Path(self.calib_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def _apply_saved_calibration(self, leader_robot) -> None:
        db = self._read_calib_db()
        port_key = str(self.gello_port)
        entry = db.get(port_key)
        if not entry or "joint_offsets" not in entry:
            raise RuntimeError(
                f"No saved calibration for port {port_key} in {self.calib_file}. "
                f"Run once with --save-calib (robot at observe, leader at home)."
            )
        joint_offsets = entry["joint_offsets"]
        leader_robot.set_joint_offsets(joint_offsets)
        self.get_logger().info(f"Applied saved calibration for {port_key} (len={len(joint_offsets)}).")

    def _save_current_calibration(self, leader_robot) -> None:
        # Save the current internal offsets after auto-calibration.
        db = self._read_calib_db()
        port_key = str(self.gello_port)
        joint_offsets = leader_robot.get_joint_offsets().tolist()
        db[port_key] = {
            "joint_offsets": joint_offsets,
            "saved_at_unix_s": time.time(),
            "note": "Fixed calibration offsets for GELLO leader. Capture with robot at observe + leader at home.",
        }
        self._write_calib_db(db)


def main():
    args = parse_args()
    rclpy.init()
    # Inject flag into node instance before __init__ runs by setting a temporary
    # attribute on the class (simple and contained).
    GelloRosTeleop._use_ros_joint_state_flag = bool(args.use_ros_joint_state)  # type: ignore[attr-defined]
    node = GelloRosTeleop(
        robot_ip=args.robot_ip,
        gello_port=args.gello_port,
        hz=args.hz,
        joint_topic=args.joint_topic,
        max_start_joint_delta=args.max_start_joint_delta,
        warmup_steps=args.warmup_steps,
        warmup_max_delta=args.warmup_max_delta,
        max_joint_step=args.max_joint_step,
        gripper_threshold=args.gripper_threshold,
        gripper_open=args.gripper_open,
        gripper_close=args.gripper_close,
        ros_joint_state_timeout=args.ros_joint_state_timeout,
        ros_joint_state_stale=args.ros_joint_state_stale,
        calib_file=args.calib_file,
        save_calib=args.save_calib,
        load_calib=args.load_calib,
        go_observe_on_start=args.go_observe_on_start,
        observe_wait_s=args.observe_wait_s,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        # Guard: Ctrl+C can race shutdown in some environments.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

