#!/usr/bin/env python3
"""
Inference script for running learned policies (SmolVLA or Octo) on a UR5 robot.

Architecture:
  rfm.robots.ur5_bridge  (runs in a separate terminal)
     ↑ /ur5/goal_joint  (JointState — absolute joint targets)
     ↓ /ur5/status       (String — "IDLE" / "MOVING")
  This script  (ROS2 node)
     - Runs model inference → produces action (absolute joint values)
     - Publishes to /ur5/goal_joint
     - Reads wrist camera via pyrealsense2 (or ROS2 topic)
     - Reads robot joint state via a separate rtde_receive connection

Usage:
  # Terminal 1 — start the bridge
  source /opt/ros/humble/setup.bash && cd /home/lcw/RFM && source venv/bin/activate
  python -m rfm.robots.ur5_bridge
  # 또는: rfm-ur5-bridge (pip install -e . 후)

  # Terminal 2 — run SmolVLA policy
  source /opt/ros/humble/setup.bash && cd /home/lcw/RFM && source venv/bin/activate
  python -m rfm.policies.runner --model-type smolvla \\
      --checkpoint outputs/train/eggplant/checkpoints/020000/pretrained_model
  # 또는: rfm-run-policy --model-type smolvla ... (pip install -e . 후)

  # Terminal 2 — run Octo policy
  python -m rfm.policies.runner --model-type octo \\
      --checkpoint /home/lcw/RFM/outputs/octo_finetune/.../5000 \\
      --task "Pick up the eggplant and place it on the plate." \\
      --window-size 2 --exec-horizon 1

  # Dry-run (model-only test — no robot, camera, or ROS2 required)
  python -m rfm.policies.runner --model-type octo --dry-run \\
      --checkpoint /home/lcw/RFM/outputs/octo_finetune/.../5000

Gripper notes:
  - In Gello training data, action[7] = gripper position (higher = more open).
    The min/max values are printed by convert_gello_to_lerobot.py during conversion.
  - The Gello gripper minimum is ~0.05 (closed), NOT 0. Sending 0 may cause some
    hardware to ignore the command entirely — use --gripper-min with the value
    from the conversion script.
  - If your hardware has inverted gripper direction, use --invert-gripper.
  - Use --calibrate-gripper to auto-detect min/max from live data.

Prerequisites:
  - ur5_rtde_bridge.py must be running before starting this script.
  - The UR5 must be in Remote Control mode.
  - Press Ctrl+C for immediate emergency stop.
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "outputs/train/eggplant/checkpoints/020000/pretrained_model"
DEFAULT_TASK = "Pick up the eggplant and place it on the plate."
DEFAULT_ROBOT_IP = "192.168.0.43"
DEFAULT_DEVICE = "cuda"
DEFAULT_FPS = 30
DEFAULT_DURATION = 200


def parse_args():
    p = argparse.ArgumentParser(
        description="Run SmolVLA / Octo policy on a UR5 robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Model selection --
    p.add_argument(
        "--model-type", type=str, default="smolvla",
        choices=["smolvla", "octo"],
        help="Policy model type (default: %(default)s)",
    )
    p.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint directory (default: %(default)s)",
    )
    p.add_argument(
        "--task", type=str, default=DEFAULT_TASK,
        help="Task description string passed to the policy (default: %(default)s)",
    )

    # -- Hardware --
    p.add_argument(
        "--robot-ip", type=str, default=DEFAULT_ROBOT_IP,
        help="UR5 robot IP for rtde_receive (default: %(default)s)",
    )
    p.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help="Torch device for SmolVLA inference (default: %(default)s)",
    )

    # -- Control loop --
    p.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="Control loop frequency in Hz (default: %(default)s)",
    )
    p.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help="Maximum execution time in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Test model inference only — no robot, camera, or ROS2 needed",
    )

    # -- Start pose --
    p.add_argument(
        "--start-pose", type=str, default="observe",
        help="Named pose to move to before starting (from ur5_saved_poses.json)",
    )
    p.add_argument(
        "--no-start-pose", action="store_true",
        help="Skip moving to a start pose",
    )

    # -- Gripper --
    p.add_argument(
        "--invert-gripper", action="store_true",
        help="Invert gripper command (use when policy: higher=open but HW is opposite)",
    )
    p.add_argument(
        "--gripper-min", type=float, default=0.0,
        help="Gripper hardware minimum (closed). Use the min from convert_gello_to_lerobot.py",
    )
    p.add_argument(
        "--gripper-max", type=float, default=1.0,
        help="Gripper hardware maximum (open). Use the max from convert_gello_to_lerobot.py",
    )
    p.add_argument(
        "--calibrate-gripper", action="store_true",
        help="Auto-detect gripper min/max by collecting values at startup",
    )
    p.add_argument(
        "--calibrate-gripper-sec", type=float, default=15.0,
        help="Duration (seconds) to collect gripper samples for auto-calibration",
    )

    # -- Camera --
    p.add_argument(
        "--use-ros2-camera", action="store_true",
        help="Subscribe to a ROS2 image topic instead of using pyrealsense2 directly",
    )
    p.add_argument(
        "--camera-topic", type=str, default="/wrist_cam/camera/color/image_raw",
        help="ROS2 image topic (only used with --use-ros2-camera)",
    )

    # -- Octo-specific --
    p.add_argument(
        "--window-size", type=int, default=2,
        help="Octo observation history window size (must match training config)",
    )
    p.add_argument(
        "--exec-horizon", type=int, default=1,
        help="Number of action-chunk steps to execute per inference (1 = most reactive)",
    )

    return p.parse_args()


# ===========================================================================
# Model loading — SmolVLA
# ===========================================================================


def load_smolvla_policy(checkpoint_path: str, device: str):
    """Load a SmolVLA policy along with its pre/post-processors.

    Args:
        checkpoint_path: Path to the pretrained model checkpoint directory.
        device: Torch device string (e.g. "cuda", "cpu").

    Returns:
        Tuple of (policy, preprocessor, postprocessor, config).
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    logger.info(f"[SmolVLA] Loading checkpoint: {checkpoint_path}")
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device

    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint_path, config=config)
    policy = policy.to(device)
    policy.eval()
    logger.info(f"[SmolVLA] Policy loaded: {config.type} on {device}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    logger.info("[SmolVLA] Pre/post-processors ready")

    return policy, preprocessor, postprocessor, config


# ===========================================================================
# Model loading — Octo
# ===========================================================================


def load_octo_policy(checkpoint_path: str):
    """Load an Octo model and wrap it in a callable policy function.

    The returned ``policy_fn`` automatically supplies a fresh JAX RNG key
    on every call (via ``supply_rng``) and un-normalizes actions using the
    dataset statistics stored in the checkpoint.

    Args:
        checkpoint_path: Path to the Octo checkpoint directory.

    Returns:
        Tuple of (model, policy_fn).
    """
    # Add local Octo source to sys.path if not pip-installed
    octo_root = Path(__file__).resolve().parent / "octo"
    if octo_root.exists() and str(octo_root) not in sys.path:
        sys.path.insert(0, str(octo_root))

    from octo.model.octo_model import OctoModel
    from octo.utils.train_callbacks import supply_rng

    logger.info(f"[Octo] Loading checkpoint: {checkpoint_path}")
    model = OctoModel.load_pretrained(checkpoint_path)

    # Log model metadata for debugging
    example_obs = model.example_batch["observation"]
    obs_keys = [k for k in example_obs.keys() if k != "timestep_pad_mask"]
    action_dim = model.example_batch["action"].shape[-1]
    logger.info(f"[Octo] Observation keys: {obs_keys}")
    logger.info(f"[Octo] Action dim: {action_dim}")
    logger.info(f"[Octo] Dataset statistics keys: {list(model.dataset_statistics.keys())}")

    # Wrap sample_actions with auto-RNG and un-normalization
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )
    logger.info("[Octo] Policy function ready")

    return model, policy_fn


# ===========================================================================
# Octo observation history
# ===========================================================================


class OctoObservationHistory:
    """Manages a sliding window of observations for Octo inference.

    Mirrors the logic of Octo's HistoryWrapper but operates without a Gym
    environment — useful for manual observation collection on real hardware.

    The window is pre-filled with copies of the first observation (zero-padded
    via ``timestep_pad_mask``), then shifts as new frames arrive.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history: deque[dict] = deque(maxlen=window_size)
        self.num_obs = 0

    def reset(self, obs: dict) -> dict:
        """Initialize the history buffer, padding with the first observation."""
        self.num_obs = 1
        self.history.clear()
        for _ in range(self.window_size):
            self.history.append(obs)
        return self._get_stacked()

    def add(self, obs: dict) -> dict:
        """Append a new observation and return the current window."""
        self.num_obs += 1
        self.history.append(obs)
        return self._get_stacked()

    def _get_stacked(self) -> dict:
        """Stack the history window and attach a timestep padding mask."""
        stacked = {
            k: np.stack([frame[k] for frame in self.history])
            for k in self.history[0]
        }
        # Mask leading frames that are copies of the first observation
        pad_length = self.window_size - min(self.num_obs, self.window_size)
        timestep_pad_mask = np.ones(self.window_size)
        timestep_pad_mask[:pad_length] = 0
        stacked["timestep_pad_mask"] = timestep_pad_mask
        return stacked


# ===========================================================================
# Camera backends
# ===========================================================================


class RealSenseCamera:
    """Direct RealSense camera access via pyrealsense2."""

    def __init__(self, width=640, height=480, fps=30, max_retries=3):
        import pyrealsense2 as rs

        # Try to stop any existing pipelines first
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            try:
                sensors = dev.query_sensors()
                for sensor in sensors:
                    sensor.stop()
            except Exception:
                pass

        # Retry initialization
        last_error = None
        for attempt in range(max_retries):
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
                self.pipeline.start(config)
                # Discard initial auto-exposure frames
                for _ in range(30):
                    self.pipeline.wait_for_frames()
                logger.info(f"RealSense camera initialized ({width}x{height} @ {fps}fps)")
                return
            except RuntimeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Camera initialization attempt {attempt + 1} failed: {e}")
                    logger.info("Waiting 2 seconds before retry...")
                    time.sleep(2)
                    # Try to clean up
                    try:
                        if hasattr(self, 'pipeline'):
                            self.pipeline.stop()
                    except Exception:
                        pass
                else:
                    # Check for running processes that might be using the camera
                    import subprocess
                    try:
                        result = subprocess.run(
                            ["ps", "aux"], capture_output=True, text=True, timeout=2
                        )
                        realsense_procs = [
                            line for line in result.stdout.split("\n")
                            if "realsense2_camera_node" in line or "realsense" in line.lower()
                        ]
                        proc_info = ""
                        if realsense_procs:
                            proc_info = "\n\n다음 프로세스가 카메라를 사용 중일 수 있습니다:\n"
                            proc_info += "\n".join(realsense_procs[:3])
                            proc_info += "\n\n해결 방법:\n"
                            proc_info += "1. 다른 사용자가 실행한 프로세스를 종료하세요:\n"
                            proc_info += "   sudo kill <PID>  # 또는 해당 사용자에게 요청\n"
                            proc_info += "2. 또는 ROS2 카메라 토픽을 사용하세요:\n"
                            proc_info += "   --use-ros2-camera --camera-topic /wrist_cam/camera/color/image_raw\n"
                    except Exception:
                        pass
                    
                    raise RuntimeError(
                        f"RealSense 카메라 초기화 실패 ({max_retries}회 시도).\n"
                        f"마지막 오류: {e}\n"
                        f"다른 프로세스가 카메라를 사용 중일 수 있습니다.{proc_info}"
                    ) from e

    def capture(self) -> np.ndarray:
        """Capture a single RGB frame. Returns (H, W, 3) uint8 array."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


class ROS2Camera:
    """Subscribes to a ROS2 image topic for camera input.

    Handles rgb8, bgr8, and mono8 encodings natively; falls back to
    cv_bridge for other formats.
    """

    def __init__(self, node, topic="/wrist_cam/camera/color/image_raw",
                 width=640, height=480, fps=30):
        """
        Args:
            node: An initialized rclpy Node instance.
            topic: ROS2 image topic to subscribe to.
            width, height, fps: Expected image dimensions (used for logging only).
        """
        from sensor_msgs.msg import Image
        from rclpy.qos import qos_profile_sensor_data

        self.node = node
        self.latest_image = None
        self.image_lock = threading.Lock()

        self.sub = self.node.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )

        # Wait for the first frame (up to ~10 seconds)
        logger.info(f"Waiting for ROS2 camera topic: {topic}")
        for _ in range(100):
            with self.image_lock:
                if self.latest_image is not None:
                    logger.info(f"ROS2 camera ready ({width}x{height})")
                    return
            import rclpy
            rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(f"Timed out waiting for images on topic: {topic}")

    def _on_image(self, msg):
        """Subscriber callback — converts the ROS2 Image to a NumPy RGB array."""
        try:
            import cv2

            if msg.encoding == "rgb8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3)
                )
            elif msg.encoding == "bgr8":
                img_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3)
                )
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "mono8":
                img_gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width)
                )
                img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            else:
                # Fallback: try cv_bridge for less common encodings
                try:
                    from cv_bridge import CvBridge
                    img = CvBridge().imgmsg_to_cv2(msg, "rgb8")
                except Exception as e:
                    logger.warning(f"Unsupported encoding '{msg.encoding}': {e}")
                    return

            with self.image_lock:
                self.latest_image = img
        except Exception as e:
            logger.warning(f"Image conversion failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def capture(self) -> np.ndarray:
        """Return the most recently received frame as an (H, W, 3) uint8 array."""
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.01)

        with self.image_lock:
            if self.latest_image is None:
                raise RuntimeError("No camera image received yet")
            return self.latest_image.copy()

    def stop(self):
        # Subscription is cleaned up when the node is destroyed
        pass


class DummyCamera:
    """Generates random noise images for dry-run testing."""

    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        logger.info(f"DummyCamera initialized ({width}x{height})")

    def capture(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def stop(self):
        pass


# ===========================================================================
# UR5 robot interfaces
# ===========================================================================


class UR5ROS2Interface:
    """Communicates with a UR5 robot through ur5_rtde_bridge.py via ROS2.

    Communication channels:
      - Joint state reading : direct rtde_receive connection (read-only, safe
                              to open alongside the bridge's own connection)
      - Joint commands      : publishes to /ur5/goal_joint (JointState)
      - Gripper commands    : publishes to /ur5/gripper_cmd (Float64)
      - Named pose movement : publishes "go <name>" to /ur5/cmd (String)
      - Status monitoring   : subscribes to /ur5/status ("IDLE" / "MOVING")
      - Emergency stop      : calls /ur5/stop service (Trigger)
    """

    def __init__(self, node, robot_ip: str):
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String, Float64
        from std_srvs.srv import Trigger

        self.node = node
        self._JointState = JointState

        # Direct RTDE connection for reading joint state
        import rtde_receive
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        logger.info(f"RTDE receive connected: {robot_ip}")

        # Publishers
        self.pub_goal_joint = node.create_publisher(JointState, "/ur5/goal_joint", 10)
        self.pub_cmd = node.create_publisher(String, "/ur5/cmd", 10)
        self.pub_gripper = node.create_publisher(Float64, "/ur5/gripper_cmd", 10)

        # Status subscriber
        self._status = "UNKNOWN"
        node.create_subscription(String, "/ur5/status", self._on_status, 10)

        # Emergency stop service client
        self.stop_client = node.create_client(Trigger, "/ur5/stop")

        logger.info("ROS2 interface initialized")

    def _on_status(self, msg):
        self._status = msg.data

    @property
    def is_idle(self) -> bool:
        return self._status == "IDLE"

    def get_joint_positions(self) -> np.ndarray:
        """Read current 6-DOF joint positions in radians."""
        return np.array(self.rtde_r.getActualQ(), dtype=np.float32)

    def get_gripper_position(self) -> float:
        """Read current gripper position. TODO: implement for your gripper hardware."""
        return 0.0

    def send_joint_command(self, joint_targets: np.ndarray):
        """Publish absolute joint targets to /ur5/goal_joint.

        The bridge receives these and executes a moveJ command.
        """
        msg = self._JointState()
        msg.position = [float(v) for v in joint_targets[:6]]
        self.pub_goal_joint.publish(msg)
        logger.debug(f"Published /ur5/goal_joint: {[f'{v:.3f}' for v in joint_targets[:6]]}")

    def send_gripper_command(self, gripper_value: float):
        """Publish a gripper position command to /ur5/gripper_cmd."""
        from std_msgs.msg import Float64
        msg = Float64()
        msg.data = float(gripper_value)
        self.pub_gripper.publish(msg)
        logger.debug(f"Published /ur5/gripper_cmd: {gripper_value:.3f}")

    def go_to_pose(self, pose_name: str):
        """Command the bridge to move to a named pose via /ur5/cmd."""
        from std_msgs.msg import String
        msg = String()
        msg.data = f"go {pose_name}"
        self.pub_cmd.publish(msg)

    def stop(self):
        """Trigger emergency stop via the /ur5/stop service."""
        from std_srvs.srv import Trigger
        if self.stop_client.wait_for_service(timeout_sec=1.0):
            future = self.stop_client.call_async(Trigger.Request())
            import rclpy
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            logger.info("Robot stopped")
        else:
            logger.warning("Stop service not available")

    def spin_once(self, timeout_sec=0.0):
        """Process pending ROS2 callbacks (e.g. status updates)."""
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)


class DummyUR5Interface:
    """Simulated UR5 interface for dry-run testing (no hardware required)."""

    def __init__(self):
        self._q = np.array([0.78, -1.49, 1.82, -1.78, -1.48, 0.10], dtype=np.float32)
        self._status = "IDLE"
        logger.info("Dummy robot initialized")

    @property
    def is_idle(self) -> bool:
        return True

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_gripper_position(self) -> float:
        return 0.05

    def send_joint_command(self, joint_targets: np.ndarray):
        self._q = joint_targets[:6].astype(np.float32)

    def send_gripper_command(self, gripper_value: float):
        pass

    def go_to_pose(self, pose_name: str):
        logger.info(f"[DRY-RUN] go {pose_name}")

    def stop(self):
        pass

    def spin_once(self, timeout_sec=0.0):
        pass


# ===========================================================================
# Observation builders
# ===========================================================================


def build_observation_smolvla(
    image: np.ndarray,
    joint_positions: np.ndarray,
    gripper_position: float,
    task: str,
) -> dict:
    """Build a SmolVLA-compatible observation dict from raw sensor data.

    The observation layout matches the Gello training data produced by
    convert_gello_to_lerobot.py:

      observation.state (22-dim) = concat([
          joint_positions  (7)   — UR5 6-DOF + Gello 7th axis
          joint_velocities (7)   — identical to joint_positions in practice
          ee_pos_quat      (7)   — zeros (not available on real hardware)
          gripper_position  (1)
      ])

      action (8-dim) = concat([
          control           (7)  — target joint values (6 UR5 + 1 Gello)
          gripper            (1)
      ])

    Args:
        image: (H, W, 3) uint8 RGB image from wrist camera.
        joint_positions: (6,) current UR5 joint positions in radians.
        gripper_position: Current gripper opening value.
        task: Natural language task description.

    Returns:
        Dict with keys expected by SmolVLA's preprocessor.
    """
    import torch

    # Image: (H, W, 3) uint8 → (1, 3, H, W) float32 [0, 1]
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    # State: 22-dim vector matching the training data layout
    jp = np.zeros(7, dtype=np.float32)
    jp[:6] = joint_positions[:6]
    jp[6] = gripper_position  # Gello 7th axis → gripper proxy

    state = np.concatenate([
        jp,                                              # joint_positions  (7)
        jp,                                              # joint_velocities (7) — same in training data
        np.zeros(7, dtype=np.float32),                   # ee_pos_quat      (7) — zeros
        np.array([gripper_position], dtype=np.float32),  # gripper           (1)
    ])  # Total: 22

    state_tensor = torch.from_numpy(state).unsqueeze(0)

    return {
        "observation.images.wrist": img_tensor,
        "observation.state": state_tensor,
        "task": [task],
    }


def build_single_obs_octo(
    image: np.ndarray,
    joint_positions: np.ndarray,
    gripper_position: float,
    model,
) -> dict:
    """Build a single-frame Octo observation dict from raw sensor data.

    The observation layout matches the RLDS training data produced by
    convert_gello_to_rlds.py:

      observation/image_0 : (480, 640, 3) uint8 → mapped to image_primary
      observation/state   : (7,) float32 — [6 joints, 1 gripper]
      action              : (7,) float32

    Images are resized to match the model's expected input resolution
    (typically 256×256, inferred from the checkpoint's example batch).

    Args:
        image: (H, W, 3) uint8 RGB image from wrist camera.
        joint_positions: (6,) current UR5 joint positions in radians.
        gripper_position: Current gripper opening value.
        model: Loaded OctoModel instance (used to infer expected shapes).

    Returns:
        Single-frame observation dict (without batch or time dimensions).
    """
    from PIL import Image as PILImage

    # Determine target resolution from the model's example batch
    example_obs = model.example_batch["observation"]
    if "image_primary" in example_obs:
        target_h, target_w = example_obs["image_primary"].shape[-3:-1]
    else:
        target_h, target_w = 256, 256

    img_pil = PILImage.fromarray(image)
    img_resized = np.array(img_pil.resize((target_w, target_h), PILImage.LANCZOS))

    obs = {"image_primary": img_resized}

    # Include proprioceptive state if the model expects it
    if "proprio" in example_obs:
        state_dim = example_obs["proprio"].shape[-1]
        state = np.zeros(state_dim, dtype=np.float32)
        state[:min(6, state_dim)] = joint_positions[:min(6, state_dim)]
        if state_dim > 6:
            state[6] = gripper_position
        obs["proprio"] = state

    return obs


# ===========================================================================
# Main control loop
# ===========================================================================


def main():
    args = parse_args()
    model_type = args.model_type

    # -- Load model --
    if model_type == "smolvla":
        policy, preprocessor, postprocessor, config = load_smolvla_policy(
            args.checkpoint, args.device
        )
    else:  # octo
        import jax
        octo_model, octo_policy_fn = load_octo_policy(args.checkpoint)
        octo_task = octo_model.create_tasks(texts=[args.task])  # Compile once
        obs_history = OctoObservationHistory(window_size=args.window_size)
        action_queue: deque = deque()
        history_initialized = False

    # -- Initialize ROS2 (needed for camera and/or robot communication) --
    ros2_node = None
    if not args.dry_run:
        import rclpy
        from rclpy.node import Node
        rclpy.init()
        ros2_node = rclpy.create_node(f"{model_type}_policy_runner")

    # -- Initialize camera --
    if args.dry_run:
        camera = DummyCamera()
    elif args.use_ros2_camera:
        camera = ROS2Camera(
            node=ros2_node, topic=args.camera_topic, width=640, height=480, fps=30
        )
    else:
        camera = RealSenseCamera(width=640, height=480, fps=30)

    # -- Initialize robot interface --
    if args.dry_run:
        robot = DummyUR5Interface()
    else:
        robot = UR5ROS2Interface(ros2_node, args.robot_ip)

        # Wait for the bridge to report IDLE status
        logger.info("Checking ur5_rtde_bridge status...")
        bridge_found = False
        for i in range(30):
            robot.spin_once(timeout_sec=0.1)
            if robot.is_idle:
                bridge_found = True
                break
            if i == 0:
                logger.info(f"  Waiting for bridge response... (status={robot._status})")

        if not bridge_found:
            logger.error("Cannot confirm bridge status!")
            logger.error("  Verify that ur5_rtde_bridge.py is running.")
            logger.error("  Debug with: ros2 topic echo /ur5/status")
            logger.warning("  Continuing anyway — robot may not respond to commands.")
        else:
            logger.info("Bridge connected (IDLE)")

        # Verify that expected ROS2 topics exist
        import subprocess
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, timeout=2,
            )
            if "/ur5/goal_joint" in result.stdout:
                logger.info("ROS2 topic verified: /ur5/goal_joint")
            else:
                logger.warning("ROS2 topic /ur5/goal_joint not found")
        except Exception as e:
            logger.warning(f"Could not verify ROS2 topics: {e}")

    # -- Move to start pose --
    if not args.dry_run and not args.no_start_pose:
        poses_path = Path(__file__).resolve().parent / "ur5_saved_poses.json"
        if poses_path.exists():
            with open(poses_path) as f:
                saved_poses = json.load(f)
            if args.start_pose in saved_poses:
                logger.info(f"Moving to start pose '{args.start_pose}'...")
                robot.go_to_pose(args.start_pose)
                time.sleep(0.5)
                for _ in range(100):
                    robot.spin_once(timeout_sec=0.1)
                    if robot.is_idle:
                        break
                time.sleep(1.0)
                logger.info("Arrived at start pose")
            else:
                logger.warning(f"Pose '{args.start_pose}' not found in ur5_saved_poses.json")
        else:
            logger.warning("ur5_saved_poses.json not found")

    # -- Signal handler for graceful shutdown --
    shutdown = False

    def signal_handler(sig, frame):
        nonlocal shutdown
        logger.info("\nShutdown signal received — stopping robot...")
        shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # -- Print execution summary --
    interval = 1.0 / args.fps
    g_min = args.gripper_min
    g_max = args.gripper_max

    logger.info("=" * 60)
    logger.info("Policy execution started")
    logger.info(f"  Model      : {model_type}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Task       : {args.task}")
    logger.info(f"  FPS        : {args.fps}, Duration: {args.duration}s")
    if model_type == "octo":
        logger.info(f"  Window     : {args.window_size}, Exec horizon: {args.exec_horizon}")
    logger.info(f"  Dry-run    : {args.dry_run}")
    logger.info(f"  Gripper    : invert={args.invert_gripper}, range=[{g_min:.2f}, {g_max:.2f}]")
    if g_min == 0.0 and not args.dry_run:
        logger.warning(
            "Gripper min is 0.0 — Gello data has min ~0.05 even when closed. "
            "Sending 0 may cause some hardware to ignore the command. "
            "Use --gripper-min with the value from convert_gello_to_lerobot.py."
        )
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    # -- Gripper auto-calibration setup --
    start_time = time.time()
    step = 0
    skipped = 0
    calibrate_gripper_samples: list[float] = []
    calibrate_end_time = (
        start_time + args.calibrate_gripper_sec
        if args.calibrate_gripper and not args.dry_run
        else 0
    )
    if calibrate_end_time > start_time:
        logger.info(
            f"Gripper auto-calibration: collecting samples for "
            f"{args.calibrate_gripper_sec:.0f}s"
        )

    # ===================================================================
    # Control loop
    # ===================================================================

    try:
        while not shutdown and (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            # Process pending ROS2 callbacks (status updates, etc.)
            robot.spin_once(timeout_sec=0.0)

            # Skip this iteration if the bridge is still executing the previous moveJ
            if not robot.is_idle:
                skipped += 1
                dt = time.perf_counter() - loop_start
                if dt < interval:
                    time.sleep(max(0, interval - dt))
                continue

            # ==========================================================
            # SmolVLA inference path
            # ==========================================================
            if model_type == "smolvla":
                import torch

                # 1. Collect observations
                image = camera.capture()
                joint_positions = robot.get_joint_positions()
                gripper_position = robot.get_gripper_position()
                if calibrate_end_time > 0 and time.time() < calibrate_end_time:
                    calibrate_gripper_samples.append(float(gripper_position))

                # 2. Build observation dict
                obs = build_observation_smolvla(
                    image, joint_positions, gripper_position, args.task
                )

                # 3. Preprocess → inference → postprocess
                processed_obs = preprocessor(obs)
                with torch.no_grad():
                    actions = policy.select_action(processed_obs)
                actions = postprocessor(actions)
                actions = actions.squeeze(0).cpu().numpy()  # (8,)

                # 4. Parse action: [control_0..6, gripper_0]
                joint_target = actions[:6]
                gripper_cmd = actions[7] if len(actions) > 7 else actions[6]

            # ==========================================================
            # Octo inference path
            # ==========================================================
            else:
                # If there are buffered actions from a previous chunk, use them
                if action_queue:
                    action = action_queue.popleft()
                    joint_target = action[:6]
                    gripper_cmd = action[6] if len(action) > 6 else 0.0
                else:
                    # 1. Collect observations
                    image = camera.capture()
                    joint_positions = robot.get_joint_positions()
                    gripper_position = robot.get_gripper_position()
                    if calibrate_end_time > 0 and time.time() < calibrate_end_time:
                        calibrate_gripper_samples.append(float(gripper_position))

                    # 2. Build single-frame observation
                    single_obs = build_single_obs_octo(
                        image, joint_positions, gripper_position, octo_model
                    )

                    # 3. Push into observation history (or initialize it)
                    if not history_initialized:
                        stacked_obs = obs_history.reset(single_obs)
                        history_initialized = True
                    else:
                        stacked_obs = obs_history.add(single_obs)

                    # 4. Add batch dimension: (window, ...) → (1, window, ...)
                    batched_obs = jax.tree.map(lambda x: x[None], stacked_obs)

                    # 5. Run model inference → (1, action_horizon, action_dim)
                    raw_actions = octo_policy_fn(batched_obs, octo_task)
                    raw_actions = np.array(raw_actions[0])  # (action_horizon, action_dim)

                    # 6. Queue exec_horizon actions from the chunk
                    for i in range(min(args.exec_horizon, len(raw_actions))):
                        action_queue.append(raw_actions[i])

                    # 7. Pop the first action
                    action = action_queue.popleft()
                    joint_target = action[:6]
                    gripper_cmd = action[6] if len(action) > 6 else 0.0

            # -- Apply gripper auto-calibration results (once) --
            if (
                calibrate_end_time > 0
                and time.time() >= calibrate_end_time
                and calibrate_gripper_samples
            ):
                g_min_hw = min(calibrate_gripper_samples)
                g_max_hw = max(calibrate_gripper_samples)
                args.gripper_min = float(g_min_hw)
                args.gripper_max = float(g_max_hw)
                logger.info(
                    f"Gripper auto-calibration applied: "
                    f"min={args.gripper_min:.3f}, max={args.gripper_max:.3f} "
                    f"({len(calibrate_gripper_samples)} samples)"
                )
                if args.gripper_min == args.gripper_max:
                    logger.warning(
                        "Gripper min == max — check get_gripper_position() implementation"
                    )
                calibrate_end_time = 0
                calibrate_gripper_samples.clear()

            # -- Scale gripper command --
            # Policy output [0, 1] → optional inversion → scale to [gripper_min, gripper_max]
            # Note: Gello data has min ~0.05 even when closed, so gripper_min=0 will
            # produce a "close" command of 0, which may be ignored by some hardware.
            g_policy = float(np.clip(gripper_cmd, 0.0, 1.0))
            g = 1.0 - g_policy if args.invert_gripper else g_policy
            g_min = args.gripper_min
            g_max = args.gripper_max
            gripper_cmd_scaled = g_min + g * (g_max - g_min)

            # -- Send commands --
            robot.send_joint_command(joint_target)
            robot.send_gripper_command(gripper_cmd_scaled)

            # -- Periodic logging --
            step += 1
            if step % 10 == 0:
                elapsed = time.time() - start_time
                extra = f" | queue={len(action_queue)}" if model_type == "octo" else ""
                gripper_display = f"gripper={gripper_cmd_scaled:.3f}"
                if g_min != 0.0 or g_max != 1.0:
                    gripper_display += f" (policy:{g_policy:.3f} -> [{g_min:.3f},{g_max:.3f}])"
                logger.info(
                    f"[Step {step:4d}] "
                    f"t={elapsed:.1f}s | "
                    f"target={np.array2string(joint_target, precision=3, suppress_small=True)} | "
                    f"{gripper_display} | "
                    f"skipped={skipped}{extra}"
                )
                skipped = 0

            # -- Maintain loop timing --
            dt = time.perf_counter() - loop_start
            sleep_time = max(0, interval - dt)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("Cleaning up...")
        robot.stop()
        camera.stop()
        if ros2_node is not None:
            ros2_node.destroy_node()
            import rclpy
            rclpy.shutdown()
        logger.info(f"Done. Executed {step} steps total.")


if __name__ == "__main__":
    main()