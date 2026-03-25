import glob
import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    """ZMQ 로봇 서버에 연결. 로봇이 다른 PC면 --hostname IP 로 지정."""

    agent: str = "none"
    robot_port: int = 6001
    # Camera client ports. If empty, no camera observations are collected/saved.
    # For 8 cameras typically use: --camera_ports 5000 5001 5002 5003 5004 5005 5006 5007
    camera_ports: Tuple[int, ...] = ()
    camera_map_file: Optional[str] = None
    hostname: str = os.environ.get("GELLO_ROBOT_HOST", "127.0.0.1")
    robot_type: Optional[str] = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    start_pose: Optional[str] = None
    pose_db_path: str = str(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "ur5_saved_poses.json")
    )
    load_gello_calib: bool = False
    gello_calib_file: str = str(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "gello_calibration.json")
    )

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    # Data save root. Use project-local folder by default for reproducibility.
    data_dir: str = "/home/lcw/RFM_lerobot/data"
    bimanual: bool = False
    verbose: bool = False
    # During the initial move to --start-pose/--start-joints, avoid blocking on camera reads.
    # This makes startup robust even if one camera is slow to deliver frames.
    move_start_without_cameras: bool = True
    # If False, observations won't include `*_depth` keys (smaller/faster dataset).
    save_depth: bool = True

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)
        if self.start_pose is not None:
            self.start_pose = str(self.start_pose).strip().lower()


def _get_robot_obs_only(robot_client: ZMQClientRobot) -> dict:
    """Get robot observations without touching camera clients."""
    return robot_client.get_observations()


def _load_pose_db_q(pose_db_path: str, name: str) -> np.ndarray:
    path = os.path.expanduser(pose_db_path)
    with open(path, "r") as f:
        db = json.load(f)
    if name not in db:
        raise ValueError(f"Pose '{name}' not found in {path}. Available: {sorted(db.keys())}")
    entry = db[name]
    if not isinstance(entry, dict) or entry.get("type") != "joint" or "q" not in entry:
        raise ValueError(f"Pose '{name}' in {path} is not a joint pose entry.")
    q = np.array(entry["q"], dtype=float)
    if q.shape != (6,):
        raise ValueError(f"Pose '{name}' has invalid q shape {q.shape}, expected (6,).")
    return q


def _load_gello_calib_offsets(calib_path: str, port: str) -> np.ndarray:
    """Load fixed GELLO calibration (joint_offsets) for a given /dev/serial/by-id/... port."""
    path = os.path.expanduser(calib_path)
    with open(path, "r") as f:
        db = json.load(f)
    if port not in db:
        raise ValueError(f"No GELLO calibration for port '{port}' in {path}. Keys: {sorted(db.keys())}")
    entry = db[port]
    if not isinstance(entry, dict) or "joint_offsets" not in entry:
        raise ValueError(f"Invalid calibration entry for port '{port}' in {path}.")
    jo = np.array(entry["joint_offsets"], dtype=float)
    if jo.ndim != 1:
        raise ValueError(f"Invalid joint_offsets shape {jo.shape} in {path} for port '{port}'.")
    return jo


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {}
        if args.camera_map_file:
            path = os.path.expanduser(str(args.camera_map_file))
            with open(path, "r") as f:
                mapping = json.load(f)
            if not isinstance(mapping, list):
                raise ValueError(f"--camera-map-file must be a JSON list: {path}")
            for ent in mapping:
                name = str(ent["name"])
                port = int(ent["port"])
                camera_clients[name] = ZMQClientCamera(port=port, host=args.hostname)
            print(f"Camera clients enabled (map): {list(camera_clients.keys())} (file={path})")
        elif len(args.camera_ports) > 0:
            for i, port in enumerate(args.camera_ports):
                name = f"cam{i}"
                camera_clients[name] = ZMQClientCamera(port=int(port), host=args.hostname)
            print(f"Camera clients enabled: {list(camera_clients.keys())} (ports={list(args.camera_ports)})")
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        include_depth=bool(args.save_depth),
    )

    agent_cfg = {}
    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": left,
                },
                "agent_right": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": right,
                },
            }
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "l",
                },
                "agent_right": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "r",
                },
            }
        elif args.agent == "spacemouse":
            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": left_path,
                    "verbose": args.verbose,
                },
                "agent_right": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": right_path,
                    "verbose": args.verbose,
                    "invert_button": True,
                },
            }
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = sorted(glob.glob("/dev/serial/by-id/*"))
                print(f"Found {len(usb_ports)} serial port(s): {usb_ports}")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"Using Gello serial port: {gello_port}")
                    if len(usb_ports) > 1:
                        print(
                            "  (여러 포트 있음 — 다른 장치 쓰려면 --gello-port /dev/serial/by-id/... 지정)"
                        )
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            agent_cfg = {
                "_target_": "gello.agents.gello_agent.GelloAgent",
                "port": gello_port,
                "start_joints": args.start_joints,
            }
            # Arm 고정 초기 자세(6축) + 그리퍼 0(닫힘) → 로봇을 먼저 여기로 보냄. 그 다음 Gello를 "지금 로봇 자세"에 캘리브레이션.
            if args.start_joints is not None:
                reset_joints = np.array(args.start_joints)
            elif args.start_pose is not None:
                q = _load_pose_db_q(args.pose_db_path, args.start_pose)
                reset_joints = np.array(list(q.tolist()) + [0.0])  # + gripper closed
            else:
                reset_joints = np.array(
                    list(np.deg2rad([0, -90, 90, -90, -90, 0])) + [0.0]
                )  # 6 arm (rad) + 1 gripper [0,1]
            # IMPORTANT: don't block on cameras during startup motion.
            curr_joints = _get_robot_obs_only(robot_client)["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = np.abs(curr_joints - reset_joints).max()
                steps = max(50, min(int(max_delta / 0.008), 600))
                step_sleep = 0.02
                if args.start_joints is not None:
                    pose_desc = "custom --start-joints"
                elif args.start_pose is not None:
                    pose_desc = f"pose '{args.start_pose}' from {os.path.expanduser(args.pose_db_path)} (gripper=0.0)"
                else:
                    pose_desc = "default pose (arm 0,-90,90,-90,-90,0 deg, gripper closed)"
                print(f"Moving robot to start pose: {pose_desc} — steps={steps}, ~{steps*step_sleep:.1f}s")
                for i, jnt in enumerate(
                    np.linspace(curr_joints, reset_joints, steps)
                ):
                    if args.move_start_without_cameras:
                        robot_client.command_joint_state(jnt)
                    else:
                        env.step(jnt)
                    time.sleep(step_sleep)
                    if (i + 1) % 50 == 0:
                        print(f"  move {i + 1}/{steps}...")
                if args.load_gello_calib:
                    print(
                        f"  → Using fixed GELLO calibration from {args.gello_calib_file} (no auto-calibration to robot pose)."
                    )
                else:
                    # 로봇이 reset_joints에 도달한 시점에, Gello를 "이 로봇 자세"에 캘리브레이션 (start_joints 전달 → DynamixelRobot이 오프셋 조정)
                    agent_cfg["start_joints"] = reset_joints
                    print("  → Gello will be calibrated to current robot pose (no need to hold Gello to match).")
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                "robot_type": args.robot_type,
                "which_hand": "l",
            }
        elif args.agent == "spacemouse":
            agent_cfg = {
                "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                "robot_type": args.robot_type,
                "verbose": args.verbose,
            }
        elif args.agent == "dummy" or args.agent == "none":
            agent_cfg = {
                "_target_": "gello.agents.agent.DummyAgent",
                "num_dofs": robot_client.num_dofs(),
            }
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    agent = instantiate_from_dict(agent_cfg)
    # Optionally apply a fixed GELLO calibration (joint_offsets) so collection is repeatable.
    if args.agent == "gello" and (not args.bimanual) and args.load_gello_calib:
        try:
            jo = _load_gello_calib_offsets(args.gello_calib_file, agent_cfg["port"])
            leader_robot = getattr(agent, "_robot", None)
            if leader_robot is None:
                raise RuntimeError("GelloAgent has no _robot attribute")
            leader_robot.set_joint_offsets(jo)
            print(f"Applied fixed GELLO calibration for port={agent_cfg['port']} (len={len(jo)})")
        except Exception as e:
            raise RuntimeError(f"Failed to load/apply GELLO calibration: {e}")
    # going to start position — 로봇은 이미 고정 초기 자세에 있음. Gello가 그 자세에 맞는지 확인.
    print("Going to start position (Gello를 로봇과 같은 자세에 맞춰 두었는지 확인)")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        print("  → 로봇은 고정 초기 자세에 있음. Gello를 그 자세에 맞춰 다시 실행하세요.")
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    from gello.utils.control_utils import SaveInterface, run_control_loop

    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=args.data_dir, agent_name=args.agent, expand_user=True
        )

    run_control_loop(env, agent, save_interface, use_colors=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
