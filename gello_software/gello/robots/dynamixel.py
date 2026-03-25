from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gello.robots.robot import Robot


class DynamixelRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        real: bool = False,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        gripper_config: Optional[Tuple[int, float, float]] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        from gello.dynamixel.driver import (
            DynamixelDriver,
            DynamixelDriverProtocol,
            FakeDynamixelDriver,
        )

        print(f"attempting to connect to port: {port}")
        self.gripper_open_close: Optional[Tuple[float, float]]
        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None

            # joint_ids.append(gripper_config[0])
            # joint_offsets.append(0.0)
            # joint_signs.append(1)
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
        else:
            self.gripper_open_close = None

        self._joint_ids = joint_ids
        self._driver: DynamixelDriverProtocol

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        if real:
            self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
            self._driver.set_torque_mode(False)
        else:
            self._driver = FakeDynamixelDriver(joint_ids)
        self._torque_on = False
        self._last_pos = None
        self._alpha = 0.99

        if start_joints is not None:
            # Calibrate: set offsets so that current reading is reported as start_joints (e.g. robot pose).
            # (raw - new_offset)*sign = start_joint  =>  new_offset = raw - start_joint*sign = old_offset + (current_joint - start_joint)*sign
            new_joint_offsets = []
            current_joints = self.get_joint_state()
            assert current_joints.shape == start_joints.shape
            start_joints_calib = np.array(start_joints)
            if gripper_config is not None:
                current_joints_arm = current_joints[:-1].copy()
                start_joints_arm = start_joints_calib[:-1].copy()
            else:
                current_joints_arm = current_joints
                start_joints_arm = start_joints_calib
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints_arm, start_joints_arm, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    joint_offset
                    + (float(c_joint) - float(s_joint)) * self._joint_signs[idx]
                )
            if gripper_config is not None:
                # Gripper: output is [0,1] from (pos[-1]-open)/(close-open). Calibrate so current -> start_joints[-1].
                c_g = current_joints[-1]
                s_g = start_joints_calib[-1]
                span = self.gripper_open_close[1] - self.gripper_open_close[0]
                new_joint_offsets.append(
                    self._joint_offsets[-1]
                    + (float(c_g) - float(s_g)) * span * self._joint_signs[-1]
                )
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos

    def get_joint_offsets(self) -> np.ndarray:
        """Return the internal joint offsets used to compute reported joint_state."""
        return self._joint_offsets.copy()

    def set_joint_offsets(self, joint_offsets: Sequence[float]) -> None:
        """Set the internal joint offsets (useful for loading a saved calibration)."""
        jo = np.array(list(joint_offsets), dtype=float)
        if jo.shape != self._joint_offsets.shape:
            raise ValueError(
                f"joint_offsets shape mismatch: expected {self._joint_offsets.shape}, got {jo.shape}"
            )
        self._joint_offsets = jo
        # Reset smoothing so we don't blend old/new coordinate frames.
        self._last_pos = None

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.set_joints((joint_state + self._joint_offsets).tolist())

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_mode(mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}
