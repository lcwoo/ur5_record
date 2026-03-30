#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for ROS teleop + camera + web + dataset recorder.
# Usage:
#   ./run_all.sh [task_name] [session_name]
# Example:
#   ./run_all.sh pick_and_place rfm_all

ROOT_DIR="/home/lcw/RFM_lerobot"
VENV="/home/lcw/RFM_lerobot/gello_software/.venv/bin/activate"
SESSION_NAME="${2:-rfm_all}"

TASK_NAME="${1:-}"
if [[ -z "${TASK_NAME}" ]]; then
  read -r -p "Task name (for dataset folder): " TASK_NAME
fi
TASK_NAME="$(echo "${TASK_NAME}" | tr ' ' '_' | tr -cd '[:alnum:]_-')"
if [[ -z "${TASK_NAME}" ]]; then
  TASK_NAME="task"
fi

RUN_TS="$(date +%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/data"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Kill with  : tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

BASE_SETUP="export ROS_DOMAIN_ID=0 && source /opt/ros/jazzy/setup.bash && cd ${ROOT_DIR} && source ${VENV}"

tmux new-session -d -s "${SESSION_NAME}" -n bridge
tmux send-keys -t "${SESSION_NAME}:bridge" \
  "${BASE_SETUP} && python -m rfm.robots.ur5_bridge --ros-args -p robot_ip:=192.168.0.44 -p use_rtde_io:=false -p force_robotiq_gripper:=true" C-m

tmux new-window -t "${SESSION_NAME}" -n camera
tmux send-keys -t "${SESSION_NAME}:camera" \
  "${BASE_SETUP} && python -m rfm.cameras.realsense_ros_multi_publisher --camera-map-file ${ROOT_DIR}/camera_ros_map.json --topic-prefix /rs --width 424 --height 240 --fps 15 --strict-profile --no-depth --start-stagger-ms 300" C-m

tmux new-window -t "${SESSION_NAME}" -n web
tmux send-keys -t "${SESSION_NAME}:web" \
  "${BASE_SETUP} && python -m rfm.web.ros_mjpeg_server --camera-map-file ${ROOT_DIR}/camera_ros_map.json --topic-prefix /rs --bind 0.0.0.0 --port 8080" C-m

tmux new-window -t "${SESSION_NAME}" -n teleop
tmux send-keys -t "${SESSION_NAME}:teleop" \
  "${BASE_SETUP} && python -m rfm.teleop.gello_ros_teleop --robot-ip 192.168.0.44 --joint-topic /ur5/servo_joint --hz 30 --max-joint-step 0.03 --use-ros-joint-state --load-calib --calib-file ${ROOT_DIR}/gello_calibration.json --go-observe-on-start --observe-wait-s 5.0" C-m

tmux new-window -t "${SESSION_NAME}" -n record
tmux send-keys -t "${SESSION_NAME}:record" \
  "${BASE_SETUP} && python -m rfm.data.ros_dataset_recorder --camera-map-file ${ROOT_DIR}/camera_ros_map.json --topic-prefix /rs --robot-joint-topic /ur5/joint_state --robot-tcp-topic /ur5/tcp_pose --teleop-joint-cmd-topic /ur5/servo_joint --teleop-gripper-cmd-topic /ur5/gripper_cmd --out-dir ${OUT_DIR} --task-name ${TASK_NAME} --hz 30 --no-depth" C-m

tmux select-window -t "${SESSION_NAME}:bridge"

echo "Started tmux session: ${SESSION_NAME}"
echo "Task name: ${TASK_NAME}"
echo "Dataset out dir: ${OUT_DIR}/${TASK_NAME}/episode_xxx/*.pkl"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Windows: bridge, camera, web, teleop, record"

