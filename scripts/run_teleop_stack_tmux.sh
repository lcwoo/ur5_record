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

SESSION_NAME="${1:-ur5_teleop}"
ROOT_DIR="/home/lcw/ur5_lerobot"
VENV="/home/lcw/ur5_lerobot/gello_software/.venv/bin/activate"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

BASE_SETUP="source /opt/ros/jazzy/setup.bash && cd ${ROOT_DIR} && source ${VENV} && export PYTHONPATH=${ROOT_DIR}/ur5:\$PYTHONPATH"

tmux new-session -d -s "${SESSION_NAME}" -n bridge
tmux send-keys -t "${SESSION_NAME}:bridge" \
  "${BASE_SETUP} && python -m ur5.robots.ur5_bridge --ros-args -p robot_ip:=192.168.0.44 -p use_rtde_io:=false -p force_robotiq_gripper:=true" C-m

tmux new-window -t "${SESSION_NAME}" -n camera
tmux send-keys -t "${SESSION_NAME}:camera" \
  "${BASE_SETUP} && python -m ur5.cameras.realsense_ros_multi_publisher \
    --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
    --topic-prefix /rs \
    --width 640 --height 480 --fps 15 \
    --no-depth --allow-fallback \
    --special-cams front_center,front_left15,front_left30,front_left45,front_right15,front_right30,front_right45,wrist \
    --special-format-order YUYV,RGB8 \
    --special-timeout-ms 8000 \
    --special-backoff-seq 1,2,4,8 \
    --start-stagger-ms 2500 \
    --start-retry-s 6.0" C-m

tmux new-window -t "${SESSION_NAME}" -n web
tmux send-keys -t "${SESSION_NAME}:web" \
  "${BASE_SETUP} && python -m ur5.web.ros_mjpeg_server \
    --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
    --topic-prefix /rs \
    --bind 0.0.0.0 --port 8080 --jpeg-quality 80 --max-fps 15" C-m

tmux select-window -t "${SESSION_NAME}:bridge"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Windows: bridge, camera, web, teleop"
