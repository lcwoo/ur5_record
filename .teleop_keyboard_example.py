#!/usr/bin/env python3
import os
import sys
import select
import termios
import time
import tty
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ur5.utils.math import rotvec_to_mat, mat_to_quat

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")

        # topics / service
        self.tcp_r_topic = self.declare_parameter("tcp__r", "/hw/goal_tcp_pose_r").value
        self.joint_r_topic = self.declare_parameter("joint_r", "/hw/goal_joint_r").value
        self.stop_service = self.declare_parameter("stop_service", "/hw/stop").value

        # steps
        self.step_lin = float(self.declare_parameter("step_lin", 0.01).value)     # meters
        self.step_rot = float(self.declare_parameter("step_rot", 0.05).value)     # radians (rotvec magnitude)
        self.step_joint = float(self.declare_parameter("step_joint", 0.1).value)   # radians
        
        # loop rate
        self.loop_hz = float(self.declare_parameter("loop_hz", 20.0).value)

        self.pub_tcp_r = self.create_publisher(PoseStamped, self.tcp_r_topic, 10)
        self.pub_joint_r = self.create_publisher(JointState, self.joint_r_topic, 10)
        self.pub_cmd = self.create_publisher(String, "/hw/cmd", 10)

        self.stop_client = self.create_client(Trigger, self.stop_service)

        self.get_logger().info(f"Publishing delta PoseStamped -> {self.tcp_r_topic}")
        self.get_logger().info("Press 'h' for help, 'SPACE' to stop, 'q' to quit.")

    def _save_current_pose(self, pose_name: str):
        name = (pose_name or "").strip()
        if not name:
            return
        msg = String()
        msg.data = f"save {name}"
        self.pub_cmd.publish(msg)

    def _publish_delta(self, dx: float, dy: float, dz: float, drv: Tuple[float, float, float]):
        delta_xyz = np.array([dx, dy, dz], dtype=np.float64)

        # rotvec -> quat
        R = rotvec_to_mat(np.array(drv, dtype=np.float64))
        qx, qy, qz, qw = mat_to_quat(R)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(delta_xyz[0])
        msg.pose.position.y = float(delta_xyz[1])
        msg.pose.position.z = float(delta_xyz[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)

        self.pub_tcp_r.publish(msg)

    def _publish_joint_delta(self, dj: Tuple[float, float, float, float, float, float]):
        msg = JointState()
        msg.position = dj
        self.pub_joint_r.publish(msg)

    def _stop(self):
        if not self.stop_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Stop service not available: {self.stop_service}")
            return
        req = Trigger.Request()
        future = self.stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.done() and future.result() is not None:
            self.get_logger().info(f"STOP: {future.result().success} ({future.result().message})")

    def _print_help(self):
        print("\n=== UR5 TELEOP KEYS ===")
        print("  w/s : -x / +x")
        print("  a/d : -y / +y")
        print("  r/f : +z / -z")
        print("  z/x : +Rx / -Rx")
        print("  c/v : +Ry / -Ry")
        print("  b/n : +Rz / -Rz")
        print("  1/2 : +base / -base")
        print("  3/4 : +shoulder / -shoulder")
        print("  5/6 : +elbow / -elbow")
        print("  7/8 : +wrist1 / -wrist1")
        print("  9/0 : +wrist2 / -wrist2")
        print("  -/= : +wrist3 / -wrist3")
        print("  SPACE: stop")
        print("  q    : quit")
        print("=======================\n")


def _get_keys(timeout_sec: float = 0.05, max_read: int = 32):
    """Return list of pressed keys (unique) available within timeout; [] if none."""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not rlist:
        return []

    keys = []
    for _ in range(max_read):
        ch = sys.stdin.read(1)
        if not ch:
            break
        keys.append(ch)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not rlist:
            break

    # de-dup while preserving order
    uniq = []
    for k in keys:
        if k not in uniq:
            uniq.append(k)
    return uniq


def main():
    rclpy.init()
    node = TeleopKeyboard()

    # terminal raw mode
    if not sys.stdin.isatty():
        node.get_logger().error("teleop_keyboard must be run in an interactive terminal (TTY).")
        node.destroy_node()
        rclpy.shutdown()
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    period = 1.0 / max(1e-3, float(node.loop_hz))

    try:
        while rclpy.ok():
            t_start = time.monotonic()
                
            rclpy.spin_once(node, timeout_sec=0.0)

            keys = _get_keys()

            # Priority handling (template publishes only one command per tick).
            if "q" in keys:
                break

            # Enter: ask pose name and publish "save <name>" to /hw/cmd
            if "\n" in keys or "\r" in keys:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                try:
                    pose_name = input("Enter pose name: ").strip()
                    print(f"{pose_name} Saved!")
                finally:
                    tty.setcbreak(fd)
                if pose_name:
                    node._save_current_pose(pose_name)
                continue

            if "h" in keys:
                node._print_help()

            if " " in keys:
                node._stop()
            dx = dy = dz = 0.0
            rvx = rvy = rvz = 0.0

            # translation
            if "w" in keys:
                dx += -node.step_lin
            if "s" in keys:
                dx += +node.step_lin
            if "a" in keys:
                dy += -node.step_lin
            if "d" in keys:
                dy += +node.step_lin
            if "r" in keys:
                dz += +node.step_lin
            if "f" in keys:
                dz += -node.step_lin

            # rotation (rotvec components, base frame)
            if "z" in keys:
                rvx += +node.step_rot
            if "x" in keys:
                rvx += -node.step_rot
            if "c" in keys:
                rvy += -node.step_rot
            if "v" in keys:
                rvy += +node.step_rot
            if "b" in keys:
                rvz += +node.step_rot
            if "n" in keys:
                rvz += -node.step_rot

       
            if "1" in keys:
                node._publish_joint_delta([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            if "2" in keys:
                node._publish_joint_delta([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            if "3" in keys:
                node._publish_joint_delta([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
            if "4" in keys:
                node._publish_joint_delta([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
            if "5" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
            if "6" in keys:
                node._publish_joint_delta([0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
            if "7" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
            if "8" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, -0.1, 0.0, 0.0])
            if "9" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
            if "0" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, 0.0, -0.1, 0.0])
            if "-" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
            if "=" in keys:
                node._publish_joint_delta([0.0, 0.0, 0.0, 0.0, 0.0, -0.1])

           
            if (dx != 0.0) or (dy != 0.0) or (dz != 0.0) or (rvx != 0.0) or (rvy != 0.0) or (rvz != 0.0):
                node._publish_delta(dx=dx, dy=dy, dz=dz, drv=(rvx, rvy, rvz))
            dt = time.monotonic() - t_start
            if dt < period:
                time.sleep(period - dt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


            
