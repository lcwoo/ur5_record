import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        include_depth: bool = True,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._include_depth = bool(include_depth)

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations: Dict[str, Any] = {}

        # Read cameras in parallel so N cameras doesn't multiply latency by N.
        # This matters a lot when a single camera is slow (USB bandwidth, exposure, etc.).
        if len(self._camera_dict) > 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _read_one(item: Tuple[str, CameraDriver]):
                name, camera = item
                image, depth = camera.read()
                return name, image, depth

            items = list(self._camera_dict.items())
            # Use up to N workers, but cap to avoid oversubscription.
            max_workers = min(len(items), 16)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_read_one, it) for it in items]
                for fut in as_completed(futs):
                    name, image, depth = fut.result()
                    observations[f"{name}_rgb"] = image
                    if self._include_depth:
                        observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
