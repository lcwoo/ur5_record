#!/usr/bin/env bash
set -euo pipefail

# Launch the full teleop stack in tmux windows:
# - ur5_bridge
# - realsense publisher
# - web mjpeg server
# - gello teleop
#
# Usage:
#   ./scripts/run_teleop_stack_tmux.sh [session_name]

SESSION_NAME="${1:-rfm_teleop}"
ROOT_DIR="/home/lcw/RFM_lerobot"
VENV="/home/lcw/RFM_lerobot/gello_software/.venv/bin/activate"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

BASE_SETUP="source /opt/ros/jazzy/setup.bash && cd ${ROOT_DIR} && source ${VENV}"

tmux new-session -d -s "${SESSION_NAME}" -n bridge
tmux send-keys -t "${SESSION_NAME}:bridge" \
  "${BASE_SETUP} && python -m rfm.robots.ur5_bridge --ros-args -p robot_ip:=192.168.0.44 -p use_rtde_io:=false -p force_robotiq_gripper:=true" C-m

tmux new-window -t "${SESSION_NAME}" -n camera
tmux send-keys -t "${SESSION_NAME}:camera" \
  "${BASE_SETUP} && python -m rfm.cameras.realsense_ros_multi_publisher --camera-map-file /home/lcw/RFM_lerobot/camera_port_map.json --topic-prefix /rs --width 424 --height 240 --fps 6 --no-depth --allow-fallback --start-stagger-ms 1000 --start-retry-s 3.0" C-m

tmux new-window -t "${SESSION_NAME}" -n web
tmux send-keys -t "${SESSION_NAME}:web" \
  "${BASE_SETUP} && python -m rfm.web.ros_mjpeg_server --camera-map-file /home/lcw/RFM_lerobot/camera_port_map.json --topic-prefix /rs --bind 0.0.0.0 --port 8080" C-m

tmux new-window -t "${SESSION_NAME}" -n teleop
tmux send-keys -t "${SESSION_NAME}:teleop" \
  "${BASE_SETUP} && python -m rfm.teleop.gello_ros_teleop --robot-ip 192.168.0.44 --joint-topic /ur5/servo_joint --hz 30 --max-joint-step 0.03 --use-ros-joint-state --load-calib --calib-file /home/lcw/RFM_lerobot/gello_calibration.json --go-observe-on-start --observe-wait-s 5.0" C-m

tmux select-window -t "${SESSION_NAME}:bridge"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Windows: bridge, camera, web, teleop"
