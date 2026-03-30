#!/usr/bin/env python3
"""UR5 RTDE Bridge — ROS2 node that connects other nodes to a UR5 robot.

This node provides a unified ROS2 interface for commanding a UR5 arm and
reading its state via the RTDE protocol. It supports both Cartesian (moveL)
and joint-space (moveJ) commands, a gripper interface with automatic fallback
from RTDE IO to RobotiqGripper (Modbus TCP), and a simple pose database for
saving/recalling named joint configurations.

Subscribed Topics (commands):
  /ur5/goal_tcp_pose   — Absolute TCP pose target (moveL)   [geometry_msgs/PoseStamped]
  /ur5/goal_tcp_pose_r — Relative TCP delta (moveL)          [geometry_msgs/PoseStamped]
  /ur5/goal_joint      — Absolute joint target (moveJ)       [sensor_msgs/JointState]
  /ur5/goal_joint_r    — Relative joint delta (moveJ)        [sensor_msgs/JointState]
  /ur5/gripper_cmd     — Gripper command                     [std_msgs/Float64]
                         value > gripper_mid → open (default gripper_mid=0: -1=close, 1=open)
  /ur5/cmd             — Text commands (see below)           [std_msgs/String]
    "where"            — Log current TCP pose + joint angles
    "list"             — Log saved pose names
    "save <name>"      — Save current joint angles as <name>
    "go <name>"        — Move (moveJ) to saved joint angles

Published Topics (state):
  /ur5/tcp_pose — Current TCP pose                           [geometry_msgs/PoseStamped]
  /ur5/status   — "IDLE" or "MOVING"                         [std_msgs/String]
                  Status is protected by a lock and published immediately on
                  transitions so that even very short motions produce at least
                  one MOVING → IDLE sequence for downstream nodes to detect.

Services:
  /ur5/stop — Emergency stop                                 [std_srvs/Trigger]

Pose Database Format (ur5_saved_poses.json):
  {
    "home": {"type": "joint", "q": [q0, q1, q2, q3, q4, q5]},
    ...
  }
"""

import json
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from ur5.utils.math import quat_to_mat, mat_to_quat, mat_to_rotvec, rotvec_to_mat


class UR5RTDEBridge(Node):
    def __init__(self):
        super().__init__("ur5_rtde_bridge")

        # ==================================================================
        # ROS2 parameters
        # ==================================================================

        self.robot_ip = self.declare_parameter("robot_ip", "192.168.0.44").value

        # moveL parameters (Cartesian space)
        self.speed_l = float(self.declare_parameter("speed_l", 0.10).value)   # m/s
        self.accel_l = float(self.declare_parameter("accel_l", 0.25).value)   # m/s²

        # moveJ parameters (joint space)
        self.speed_j = float(self.declare_parameter("speed_j", 1.0).value)    # rad/s
        self.accel_j = float(self.declare_parameter("accel_j", 1.0).value)    # rad/s²

        self.publish_rate = float(self.declare_parameter("publish_rate", 30.0).value)

        # servoJ parameters (streaming joint control)
        # Note: This is the mode you want for smooth teleop. It sends continuous
        # joint targets rather than discrete moveJ motions.
        self.servo_hz = float(self.declare_parameter("servo_hz", 100.0).value)
        self.servo_lookahead_time = float(self.declare_parameter("servo_lookahead_time", 0.1).value)
        self.servo_gain = int(self.declare_parameter("servo_gain", 300).value)

        # Gripper threshold: values above this are treated as "open".
        # Use 0.0 for a [-1, 1] command range, or 0.5 for a [0, 1] range.
        self.gripper_mid = float(self.declare_parameter("gripper_mid", 0.0).value)

        # Invert gripper direction (use when hardware polarity is opposite to
        # the Gello training data convention where higher = more open).
        self.invert_gripper = self.declare_parameter("invert_gripper", False).value

        # Gripper hardware range. Gello training data typically uses ~0.047–0.772.
        # These defaults match the Gello range; override if your hardware differs.
        self.gripper_min_hw = self.declare_parameter("gripper_min_hw", 0.0471).value
        self.gripper_max_hw = self.declare_parameter("gripper_max_hw", 0.7725).value

        # RTDE IO toggle. Disable if EtherNet/IP, PROFINET, or MODBUS is already
        # using the IO registers — the bridge will still work without gripper control.
        self.use_rtde_io = self.declare_parameter("use_rtde_io", True).value

        # RobotiqGripper fallback (Modbus TCP on port 63352). Activates automatically
        # when RTDE IO is unavailable and this parameter is True.
        self.use_robotiq_gripper = self.declare_parameter("use_robotiq_gripper", True).value
        
        # Force RobotiqGripper even when RTDE IO is available (useful when RTDE IO
        # doesn't actually control the gripper hardware)
        self.force_robotiq_gripper = self.declare_parameter("force_robotiq_gripper", False).value

        # ==================================================================
        # RTDE connections
        # ==================================================================

        import rtde_control, rtde_receive, rtde_io
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_io = None
        self.robotiq_gripper = None

        if self.use_rtde_io:
            try:
                self.rtde_io = rtde_io.RTDEIOInterface(self.robot_ip)
                self.get_logger().info("RTDE IO enabled — gripper via Tool DO 0")
            except RuntimeError as e:
                if "already in use" in str(e) or "RTDE" in str(e):
                    self.get_logger().warn(
                        "RTDE IO unavailable (EtherNet/IP, PROFINET, or MODBUS may "
                        "be occupying the registers). Will attempt RobotiqGripper fallback."
                    )
                else:
                    raise
        else:
            self.get_logger().info("RTDE IO disabled (use_rtde_io=false)")

        # -- RobotiqGripper fallback (always try to initialize, even if RTDE IO is available) --
        self._init_robotiq_gripper_fallback()

        # ==================================================================
        # Thread-safe motion state
        # ==================================================================

        self._state_lock = threading.Lock()
        self._moving: bool = False
        self._status: str = "IDLE"

        # ==================================================================
        # Publishers
        # ==================================================================

        self.pub_tcp_pose = self.create_publisher(PoseStamped, "/ur5/tcp_pose", 10)
        # Joint state (used by teleop adapters to avoid opening an extra RTDEReceive connection)
        self.pub_joint_state = self.create_publisher(JointState, "/ur5/joint_state", 10)
        self.pub_status = self.create_publisher(String, "/ur5/status", 10)

        # ==================================================================
        # Subscribers — Cartesian commands (moveL)
        # ==================================================================

        self.sub_tcp_abs = self.create_subscription(
            PoseStamped, "/ur5/goal_tcp_pose", self._on_tcp_abs, 10,
        )
        self.sub_tcp_rel = self.create_subscription(
            PoseStamped, "/ur5/goal_tcp_pose_r", self._on_tcp_rel, 10,
        )

        # ==================================================================
        # Subscribers — Joint commands (moveJ)
        # ==================================================================

        self.sub_joint_abs = self.create_subscription(
            JointState, "/ur5/goal_joint", self._on_joint_abs, 10,
        )
        self.sub_joint_rel = self.create_subscription(
            JointState, "/ur5/goal_joint_r", self._on_joint_rel, 10,
        )

        # Streaming joint servo (servoJ). This enables smooth teleop.
        self.sub_joint_servo = self.create_subscription(
            JointState, "/ur5/servo_joint", self._on_joint_servo, 10,
        )

        # Latest servo target (thread-safe via _state_lock)
        self._servo_target_q: Optional[Sequence[float]] = None
        self._servo_active: bool = False
        self._servo_last_ts: float = 0.0
        self._servo_timeout_s: float = 0.5  # if no updates arrive, stop servoing
        self._servo_err_last_log_ts: float = 0.0

        if self.servo_hz > 0:
            self.create_timer(1.0 / self.servo_hz, self._servo_tick)

        # ==================================================================
        # Subscribers — Gripper & text commands
        # ==================================================================

        self.sub_gripper = self.create_subscription(
            Float64, "/ur5/gripper_cmd", self._on_gripper_cmd, 10,
        )
        self.sub_cmd = self.create_subscription(
            String, "/ur5/cmd", self._on_cmd, 10,
        )

        # ==================================================================
        # Service — Emergency stop
        # ==================================================================

        self.srv_stop = self.create_service(Trigger, "/ur5/stop", self._on_stop)

        # ==================================================================
        # Pose database (JSON file on disk)
        # ==================================================================

        # Use repository-root pose DB so it is shared with other tools/scripts.
        repo_root = Path(__file__).resolve().parents[2]
        self.pose_db_path = repo_root / "ur5_saved_poses.json"
        self.pose_db: Dict[str, Dict] = {}
        self._load_pose_db()

        # ==================================================================
        # Periodic state publisher
        # ==================================================================

        if self.publish_rate > 0:
            self.create_timer(1.0 / self.publish_rate, self._publish_state)

        # -- Startup summary --
        self.get_logger().info(
            f"UR5 RTDE bridge connected to {self.robot_ip}. "
            f"moveL: speed={self.speed_l}, accel={self.accel_l} | "
            f"moveJ: speed={self.speed_j}, accel={self.accel_j}"
        )
        self.get_logger().info(
            "Subscribing: /ur5/goal_tcp_pose, /ur5/goal_tcp_pose_r, "
            "/ur5/goal_joint, /ur5/goal_joint_r, /ur5/servo_joint, /ur5/gripper_cmd"
        )
        self.get_logger().info("Cmd topic: /ur5/cmd (where / list / save / go)")
        self.get_logger().info("Publishing: /ur5/tcp_pose, /ur5/joint_state, /ur5/status")

    # ======================================================================
    # Initialization helpers
    # ======================================================================

    def _init_robotiq_gripper_fallback(self):
        """Attempt to connect a RobotiqGripper as fallback when RTDE IO is unavailable.
        
        Note: RobotiqGripper is also initialized even when RTDE IO is available,
        so it can be used as a fallback if RTDE IO commands don't work.
        """
        if not self.use_robotiq_gripper:
            self.get_logger().info("RobotiqGripper fallback disabled (use_robotiq_gripper=false)")
            return

        self.get_logger().info("RTDE IO unavailable — attempting RobotiqGripper connection...")

        try:
            # gello_software lives at the repository root.
            gello_path = Path(__file__).resolve().parents[2] / "gello_software"
            self.get_logger().info(f"Looking for gello_software at: {gello_path}")
            if not gello_path.exists():
                self.get_logger().error(
                    f"gello_software not found at {gello_path}. "
                    "RobotiqGripper fallback unavailable."
                )
                return

            sys.path.insert(0, str(gello_path))
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.get_logger().info(f"Connecting RobotiqGripper at {self.robot_ip}:63352...")
            self.robotiq_gripper = RobotiqGripper()
            self.robotiq_gripper.connect(hostname=self.robot_ip, port=63352)
            self.get_logger().info("RobotiqGripper connected (Modbus TCP port 63352)")

            # Activate the gripper (skip auto-calibration to save time)
            self.get_logger().info("Activating RobotiqGripper...")
            self.robotiq_gripper.activate(auto_calibrate=False)
            if self.robotiq_gripper.is_active():
                self.get_logger().info("RobotiqGripper activated successfully")
            else:
                self.get_logger().warn("RobotiqGripper activation may have failed — check status")

        except ImportError as e:
            self.get_logger().error(f"Failed to import RobotiqGripper module: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.robotiq_gripper = None
        except Exception as e:
            self.get_logger().error(f"RobotiqGripper connection failed: {e}")
            self.get_logger().error(
                f"Verify that a Robotiq gripper is reachable at {self.robot_ip}:63352"
            )
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.robotiq_gripper = None

    # ======================================================================
    # Status helpers (thread-safe)
    # ======================================================================

    def _get_status(self) -> str:
        with self._state_lock:
            return self._status

    def _set_status(self, status: str):
        with self._state_lock:
            self._status = status
        self._publish_status(status)

    def _publish_status(self, status: Optional[str] = None):
        msg = String()
        msg.data = status or self._get_status()
        try:
            self.pub_status.publish(msg)
        except Exception:
            pass  # Middleware may not be ready yet

    # ======================================================================
    # Periodic state publisher
    # ======================================================================

    def _publish_state(self):
        """Publish current TCP pose, joint state, and status at a fixed rate."""
        try:
            pose = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            x, y, z, rx, ry, rz = [float(v) for v in pose]
            R = rotvec_to_mat(np.array([rx, ry, rz], dtype=np.float64))
            qx, qy, qz, qw = mat_to_quat(R)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)
            msg.pose.orientation.w = float(qw)
            self.pub_tcp_pose.publish(msg)

            # Joint state (6-DoF UR arm joints, radians)
            q = self.rtde_r.getActualQ()
            jmsg = JointState()
            jmsg.header.stamp = msg.header.stamp
            # Names are optional for consumers; provide a conventional ordering.
            jmsg.name = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            jmsg.position = [float(v) for v in q]
            self.pub_joint_state.publish(jmsg)

            self._publish_status()
        except Exception:
            pass  # Silently skip if the robot connection is temporarily unstable

    # ======================================================================
    # Motion execution (thread-safe, non-blocking)
    # ======================================================================

    def _start_motion(self, worker_fn: Callable[[], object], busy_msg: str) -> bool:
        """Run a blocking RTDE motion command in a background thread.

        Publishes MOVING status immediately when a motion is accepted, and
        IDLE when the motion completes (or fails). Only one motion can run
        at a time; concurrent requests are rejected with a warning.

        Args:
            worker_fn: Callable that performs the blocking RTDE command.
            busy_msg: Warning message logged when a motion is already in progress.

        Returns:
            True if the motion was accepted, False if rejected (already moving).
        """
        with self._state_lock:
            if self._moving:
                self.get_logger().warn(busy_msg)
                return False
            self._moving = True

        # Publish MOVING immediately so downstream nodes see the transition
        self._set_status("MOVING")

        def runner():
            try:
                result = worker_fn()
                # RTDE motion APIs may return False without raising.
                if result is False:
                    self.get_logger().error(
                        "Motion command returned False (robot did not execute the move). "
                        "Check Remote mode / program run state / safety status."
                    )
                    self._log_rtde_state_snapshot()
            except Exception as e:
                self.get_logger().error(f"Motion failed: {e}")
                self._log_rtde_state_snapshot()
            finally:
                with self._state_lock:
                    self._moving = False
                self._set_status("IDLE")

        threading.Thread(target=runner, daemon=True).start()
        return True

    def _log_rtde_state_snapshot(self):
        """Best-effort RTDE state dump to diagnose silent motion rejections."""
        parts = []
        checks = [
            ("isProgramRunning", "program_running"),
            ("isProtectiveStopped", "protective_stopped"),
            ("isEmergencyStopped", "emergency_stopped"),
            ("isSteady", "is_steady"),
            ("getRobotMode", "robot_mode"),
            ("getSafetyMode", "safety_mode"),
        ]
        for method_name, label in checks:
            fn = getattr(self.rtde_r, method_name, None)
            if fn is None:
                continue
            try:
                parts.append(f"{label}={fn()}")
            except Exception:
                continue
        if parts:
            self.get_logger().warn("[RTDE state] " + ", ".join(parts))

    # ======================================================================
    # Text command handler (/ur5/cmd)
    # ======================================================================

    def _on_cmd(self, msg: String):
        """Handle text commands: where, list, save <name>, go <name>."""
        line = (msg.data or "").strip()
        if not line:
            return

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip().lower() if len(parts) == 2 else None

        if cmd == "where":
            tcp = self.rtde_r.getActualTCPPose()
            q = self.rtde_r.getActualQ()
            self.get_logger().info(f"Current TCP: {tcp}")
            self.get_logger().info(f"Current Q  : {q}")
            return

        if cmd == "list":
            keys = sorted(self.pose_db.keys())
            self.get_logger().info(f"Saved poses: {keys}")
            return

        if cmd == "save":
            if not arg:
                self.get_logger().warn("Usage: save <name>")
                return
            q = self.rtde_r.getActualQ()
            entry = {"type": "joint", "q": [float(v) for v in q]}
            self.pose_db[arg] = entry
            self._save_pose_db()
            self.get_logger().info(f"Saved '{arg}' (joint angles)")
            return

        if cmd == "go":
            if not arg:
                self.get_logger().warn("Usage: go <name>")
                return
            self._go_saved_joint(arg)
            return

        self.get_logger().warn(
            f"Unknown command '{cmd}'. Supported: where, list, save <name>, go <name>"
        )

    # ======================================================================
    # Gripper command handler (/ur5/gripper_cmd)
    # ======================================================================

    def _on_gripper_cmd(self, msg: Float64):
        """Process a gripper command, dispatching to RTDE IO or RobotiqGripper.

        The command value is first optionally inverted (--invert-gripper), then
        compared against ``gripper_mid`` to determine open/close for RTDE IO,
        or normalized to [0, 255] for the RobotiqGripper.
        """
        v = float(msg.data)

        # Apply inversion if hardware polarity is opposite to the policy convention
        if self.invert_gripper:
            v = 1.0 - v

        # Determine open/close for the RTDE IO binary interface
        is_open = v > self.gripper_mid

        # --- Try RTDE IO first (but allow fallback if RobotiqGripper is available) ---
        rtde_io_success = False
        if self.rtde_io is not None and not self.force_robotiq_gripper:
            try:
                self.rtde_io.setToolDigitalOut(0, is_open)
                self.get_logger().info(f"Gripper cmd: {v:.3f} -> Tool DO 0 = {is_open}")
                rtde_io_success = True
                # If RobotiqGripper is also available, log that RTDE IO was used
                if self.robotiq_gripper is not None:
                    self.get_logger().debug("RTDE IO used (RobotiqGripper available as fallback)")
                # RTDE IO succeeded; nothing else to do.
                return
            except Exception as e:
                self.get_logger().warn(f"RTDE IO gripper command failed: {e}")

        # --- Fall back to RobotiqGripper if RTDE IO failed, not available, or forced ---
        if (not rtde_io_success or self.force_robotiq_gripper) and self.robotiq_gripper is not None:
            try:
                # Convert the command value to a 0–255 integer position.
                # If hardware range is configured, normalize within that range first.
                if self.gripper_min_hw is not None and self.gripper_max_hw is not None:
                    # v is already scaled to the hardware range (e.g. 0.047–0.772)
                    # by run_policy_ur5.py; normalize to [0, 1] then scale to [0, 255].
                    v_norm = (v - self.gripper_min_hw) / (self.gripper_max_hw - self.gripper_min_hw)
                    v_norm = max(0.0, min(1.0, v_norm))
                    pos = int(v_norm * 255)
                else:
                    # Assume v is in [0, 1] range
                    pos = int(max(0.0, min(1.0, v)) * 255)

                pos = max(0, min(255, pos))
                self.robotiq_gripper.move(pos, 255, 10)  # (position, speed, force)

                invert_tag = " (inverted)" if self.invert_gripper else ""
                range_tag = (
                    f" [{self.gripper_min_hw:.3f}–{self.gripper_max_hw:.3f}]"
                    if self.gripper_min_hw is not None else ""
                )
                self.get_logger().info(
                    f"Gripper cmd: {msg.data:.3f} -> RobotiqGripper pos={pos}{invert_tag}{range_tag}"
                )
                return
            except Exception as e:
                self.get_logger().warn(f"RobotiqGripper command failed: {e}")

        # --- Neither backend available ---
        self.get_logger().warn(
            f"Gripper command received but no backend available "
            f"(rtde_io={'OK' if self.rtde_io else 'None'}, "
            f"robotiq={'OK' if self.robotiq_gripper else 'None'})"
        )

    # ======================================================================
    # Named pose recall
    # ======================================================================

    def _go_saved_joint(self, name: str):
        """Move to a named joint pose from the pose database."""
        key = name.lower()
        if key not in self.pose_db:
            self.get_logger().warn(
                f"No saved pose '{key}'. Use 'list' to see available poses."
            )
            return

        entry = self.pose_db[key]
        if not isinstance(entry, dict) or entry.get("type") != "joint" or "q" not in entry:
            self.get_logger().warn(
                f"Saved entry '{key}' is not a valid joint pose (file may be legacy format)."
            )
            return

        q = entry["q"]
        if not (isinstance(q, list) and len(q) == 6):
            self.get_logger().warn(f"Saved joint pose '{key}' has invalid data.")
            return

        self.get_logger().info(f"Moving to pose '{key}' (moveJ)")
        self._start_motion(
            lambda: self.rtde_c.moveJ(q, speed=self.speed_j, acceleration=self.accel_j),
            busy_msg="Robot is moving — ignoring 'go' command.",
        )

    # ======================================================================
    # Pose database I/O
    # ======================================================================

    def _load_pose_db(self):
        """Load saved poses from the JSON file on disk."""
        try:
            if not self.pose_db_path.exists():
                self.get_logger().info(
                    f"Pose database not found at {self.pose_db_path} (starting empty)"
                )
                return

            with open(self.pose_db_path, "r") as f:
                data = json.load(f) or {}

            if not isinstance(data, dict):
                self.get_logger().warn("Pose database root is not a dict — ignoring file")
                return

            loaded = {}
            for k, v in data.items():
                name = str(k).strip().lower()
                if not name:
                    continue
                loaded[name] = {"type": "joint", "q": [float(x) for x in v["q"]]}

            self.pose_db = loaded
            self.get_logger().info(
                f"Loaded pose database: {self.pose_db_path} "
                f"(poses: {list(self.pose_db.keys())})"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to load pose database: {e}")

    def _save_pose_db(self):
        """Persist the current pose database to disk."""
        try:
            os.makedirs(self.pose_db_path.parent, exist_ok=True)
            with open(self.pose_db_path, "w") as f:
                json.dump(self.pose_db, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to save pose database: {e}")

    # ======================================================================
    # Emergency stop service (/ur5/stop)
    # ======================================================================

    def _on_stop(self, req, resp):
        """Best-effort emergency stop. Calls stopL (and stopJ if available)."""
        try:
            self.rtde_c.stopL(0.5)
            if hasattr(self.rtde_c, "stopJ"):
                try:
                    self.rtde_c.stopJ(0.5)
                except Exception:
                    pass
            resp.success = True
            resp.message = "Stop command sent"
        except Exception as e:
            resp.success = False
            resp.message = str(e)

        self._publish_status()
        return resp

    # ======================================================================
    # moveL callbacks (Cartesian space)
    # ======================================================================

    @staticmethod
    def _pose_to_rtde_target(msg: PoseStamped):
        """Convert a ROS2 PoseStamped (position + quaternion) to UR's [x, y, z, rx, ry, rz] format."""
        p = msg.pose.position
        q = msg.pose.orientation
        R = quat_to_mat([q.x, q.y, q.z, q.w])
        rv = mat_to_rotvec(R)
        return [float(p.x), float(p.y), float(p.z),
                float(rv[0]), float(rv[1]), float(rv[2])]

    @staticmethod
    def _unwrap_rotvec_near(rotvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return an equivalent rotation vector close to *reference*.

        Rotation vectors (axis-angle) are not unique: the same rotation can be
        represented as ``v + 2πk · axis`` for any integer k. UR's moveL
        interpolates linearly in 6D pose space, so a sudden wrap near ±π can
        cause the robot to spin in an unexpected direction.

        This function picks the representation that minimizes the Euclidean
        distance to the current (reference) rotation vector, ensuring smooth
        interpolation.
        """
        v = np.asarray(rotvec, dtype=np.float64).reshape(3)
        ref = np.asarray(reference, dtype=np.float64).reshape(3)

        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            return v

        axis = v / theta

        # Find integer k that minimizes ‖(v + 2πk·axis) − ref‖
        k0 = int(np.round((float(np.dot(axis, ref)) - theta) / (2.0 * np.pi)))

        best = v
        best_norm = float(np.linalg.norm(v - ref))
        for k in (k0 - 1, k0, k0 + 1):
            candidate = v + (2.0 * np.pi * float(k)) * axis
            dist = float(np.linalg.norm(candidate - ref))
            if dist < best_norm:
                best = candidate
                best_norm = dist

        return best

    def _on_tcp_abs(self, msg: PoseStamped):
        """Handle absolute TCP pose command (moveL)."""
        target = self._pose_to_rtde_target(msg)
        self._start_motion(
            lambda: self.rtde_c.moveL(target, speed=self.speed_l, acceleration=self.accel_l),
            busy_msg="Robot is moving — ignoring /ur5/goal_tcp_pose.",
        )

    def _on_tcp_rel(self, msg: PoseStamped):
        """Handle relative TCP delta command (moveL).

        Important: UR represents TCP orientation as a rotation vector (axis-angle).
        Rotation vectors do NOT compose by simple addition. Instead, we:
          1. Convert the delta quaternion to a rotation matrix R_delta.
          2. Multiply on the left for a base-frame relative rotation:
             R_target = R_delta @ R_cur
             (Use R_cur @ R_delta instead for tool-frame rotation.)
          3. Convert back to a rotation vector and unwrap to avoid 2π jumps.
        """
        # Delta translation (base frame)
        p = msg.pose.position
        delta_xyz = np.array([p.x, p.y, p.z], dtype=np.float64)

        # Delta rotation (quaternion → rotation matrix)
        q = msg.pose.orientation
        q_delta = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        qn = float(np.linalg.norm(q_delta))
        if qn < 1e-12:
            q_delta = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q_delta = q_delta / qn
        R_delta = quat_to_mat([float(q_delta[i]) for i in range(4)])

        def worker():
            pose = self.rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            x, y, z, rx, ry, rz = [float(v) for v in pose]

            # Apply translational delta in base frame
            x_t = x + float(delta_xyz[0])
            y_t = y + float(delta_xyz[1])
            z_t = z + float(delta_xyz[2])

            # Compose rotations (left-multiply for base-frame rotation)
            rv_cur = np.array([rx, ry, rz], dtype=np.float64)
            R_cur = rotvec_to_mat(rv_cur)
            R_target = R_delta @ R_cur
            rv_target = mat_to_rotvec(R_target)

            # Unwrap to avoid discontinuous jumps near ±π
            rv_target = self._unwrap_rotvec_near(rv_target, rv_cur)

            target = [
                float(x_t), float(y_t), float(z_t),
                float(rv_target[0]), float(rv_target[1]), float(rv_target[2]),
            ]
            return self.rtde_c.moveL(target, speed=self.speed_l, acceleration=self.accel_l)

        self._start_motion(worker, busy_msg="Robot is moving — ignoring /ur5/goal_tcp_pose_r.")

    # ======================================================================
    # moveJ callbacks (joint space)
    # ======================================================================

    @staticmethod
    def _joint_from_msg(msg: JointState) -> Optional[Sequence[float]]:
        """Extract 6 joint values from a JointState message, or None if invalid."""
        if not msg.position or len(msg.position) < 6:
            return None
        return [float(x) for x in msg.position[:6]]

    def _on_joint_abs(self, msg: JointState):
        """Handle absolute joint target command (moveJ)."""
        q = self._joint_from_msg(msg)
        if q is None:
            self.get_logger().warn(
                "/ur5/goal_joint requires JointState.position with at least 6 values."
            )
            return
        self.get_logger().info(
            f"Received /ur5/goal_joint: {[f'{v:.3f}' for v in q]} -> executing moveJ"
        )
        self._start_motion(
            lambda: self.rtde_c.moveJ(q, speed=self.speed_j, acceleration=self.accel_j),
            busy_msg="Robot is moving — ignoring /ur5/goal_joint.",
        )

    def _on_joint_rel(self, msg: JointState):
        """Handle relative joint delta command (moveJ)."""
        dq = self._joint_from_msg(msg)
        if dq is None:
            self.get_logger().warn(
                "/ur5/goal_joint_r requires JointState.position with at least 6 values."
            )
            return

        def worker():
            q = self.rtde_r.getActualQ()
            q_target = [float(q[i]) + float(dq[i]) for i in range(6)]
            return self.rtde_c.moveJ(q_target, speed=self.speed_j, acceleration=self.accel_j)

        self._start_motion(worker, busy_msg="Robot is moving — ignoring /ur5/goal_joint_r.")

    # ======================================================================
    # servoJ callbacks (streaming joint space)
    # ======================================================================

    def _on_joint_servo(self, msg: JointState):
        """Handle streaming joint target (servoJ)."""
        q = self._joint_from_msg(msg)
        if q is None:
            self.get_logger().warn(
                "/ur5/servo_joint requires JointState.position with at least 6 values."
            )
            return
        now = self.get_clock().now().nanoseconds / 1e9
        with self._state_lock:
            self._servo_target_q = q
            self._servo_last_ts = float(now)
            self._servo_active = True

    def _servo_tick(self):
        """Periodic servoJ streaming loop for smooth teleop."""
        try:
            with self._state_lock:
                if not self._servo_active or self._servo_target_q is None:
                    return
                age = (self.get_clock().now().nanoseconds / 1e9) - float(self._servo_last_ts)
                q = list(self._servo_target_q)

            # If commands stop coming, stop servo motion gently.
            if age > self._servo_timeout_s:
                with self._state_lock:
                    self._servo_active = False
                try:
                    self.rtde_c.stopJ(0.5)
                except Exception:
                    pass
                return

            # servoJ signature in rtde_control:
            # servoJ(q, speed, acceleration, time, lookahead_time, gain)
            dt = 1.0 / max(1e-6, self.servo_hz)
            # Some rtde_control builds don't accept keyword args; use positional.
            # Also use initPeriod()/waitPeriod() for stable cycle timing when available.
            try:
                t_start = self.rtde_c.initPeriod()
            except Exception:
                t_start = None

            self.rtde_c.servoJ(
                q,
                float(self.speed_j),
                float(self.accel_j),
                float(dt),
                float(self.servo_lookahead_time),
                int(self.servo_gain),
            )

            if t_start is not None:
                try:
                    self.rtde_c.waitPeriod(t_start)
                except Exception:
                    pass
        except Exception as e:
            # Avoid crashing the node if robot comm blips during streaming,
            # but log occasionally so "not moving" isn't silent.
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - float(self._servo_err_last_log_ts)) > 2.0:
                self._servo_err_last_log_ts = float(now)
                self.get_logger().warn(f"servoJ tick failed (will retry): {e}")


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    rclpy.init()
    node = UR5RTDEBridge()
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