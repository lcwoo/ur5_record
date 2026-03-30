# UR5 Teleop/Data Stack

UR5 + GELLO + RealSense 기반 데이터 수집/실행 레포입니다.

## 빠른 시작 (실사용)

## tmux로 한 번에 실행

```bash
cd /home/lcw/ur5_lerobot
./scripts/run_teleop_stack_tmux.sh ur5_teleop
tmux attach -t ur5_teleop

source /opt/ros/jazzy/setup.bash
cd /home/lcw/ur5_lerobot
source /home/lcw/ur5_lerobot/gello_software/.venv/bin/activate
python -m ur5.data.ros_dataset_recorder \
  --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
  --topic-prefix /rs \
  --robot-joint-topic /ur5/joint_state \
  --robot-tcp-topic /ur5/tcp_pose \
  --teleop-joint-cmd-topic /ur5/servo_joint \
  --teleop-gripper-cmd-topic /ur5/gripper_cmd \
  --out-dir /home/lcw/ur5_lerobot/data \
  --task-name task_id \
  --hz 30 \
  --no-depth
  
```

창 이름:
- `bridge`
- `camera`
- `web`
- `teleop`

주의/권장:
- realsense-viewer는 실행 전에 닫아 주세요(디바이스 락 방지).
- 카메라는 YUYV→RGB8 순으로 안정적으로 붙도록 설정되어 있습니다.
- 필요 시 웹은 `http://<THIS_PC_IP>:8080/`에서 확인합니다.

### 1) UR5 bridge
```bash
source /opt/ros/jazzy/setup.bash
cd /home/lcw/ur5_lerobot
source /home/lcw/ur5_lerobot/gello_software/.venv/bin/activate
python -m ur5.robots.ur5_bridge --ros-args \
  -p robot_ip:=192.168.0.44 \
  -p use_rtde_io:=false \
  -p force_robotiq_gripper:=true
```

### 2) RealSense publisher (8 cams)
```bash
source /opt/ros/jazzy/setup.bash
cd /home/lcw/ur5_lerobot
source /home/lcw/ur5_lerobot/gello_software/.venv/bin/activate
export PYTHONPATH=/home/lcw/ur5_lerobot/ur5:$PYTHONPATH
python -m ur5.cameras.realsense_ros_multi_publisher \
  --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
  --topic-prefix /rs \
  --width 640 --height 480 --fps 15 \
  --no-depth --allow-fallback \
  --special-cams front_center,front_left15,front_left30,front_left45,front_right15,front_right30,front_right45,wrist \
  --special-format-order YUYV,RGB8 \
  --special-timeout-ms 8000 \
  --special-backoff-seq 1,2,4,8 \
  --start-stagger-ms 2500 \
  --start-retry-s 6.0
```

참고: `wrist(D405)`는 환경에 따라 `424x240@5`로 fallback될 수 있습니다.

### 3) (옵션) 카메라 웹 뷰어
```bash
source /opt/ros/jazzy/setup.bash
cd /home/lcw/ur5_lerobot
source /home/lcw/ur5_lerobot/gello_software/.venv/bin/activate
export PYTHONPATH=/home/lcw/ur5_lerobot/ur5:$PYTHONPATH
python -m ur5.web.ros_mjpeg_server \
  --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
  --topic-prefix /rs \
  --bind 0.0.0.0 --port 8080 --jpeg-quality 80 --max-fps 15
```
브라우저: `http://<THIS_PC_IP>:8080/`

### 4) GELLO teleop (servo + fixed calib + observe start)
```bash
source /opt/ros/jazzy/setup.bash
   /home/lcw/ur5_lerobot/gello_software/.venv/bin/python -m ur5.teleop.gello_ros_teleop \
     --robot-ip 192.168.0.44 \
     --joint-topic /ur5/servo_joint \
     --hz 30 \
     --max-joint-step 0.03 \
     --use-ros-joint-state \
     --load-calib \
     --calib-file /home/lcw/ur5_lerobot/gello_calibration.json \
     --go-observe-on-start \
     --observe-wait-s 5.0
```

### 5) ROS dataset recorder
```bash
source /opt/ros/jazzy/setup.bash
cd /home/lcw/ur5_lerobot
source /home/lcw/ur5_lerobot/gello_software/.venv/bin/activate
python -m ur5.data.ros_dataset_recorder \
  --camera-map-file /home/lcw/ur5_lerobot/camera_port_map.json \
  --topic-prefix /rs \
  --robot-joint-topic /ur5/joint_state \
  --robot-tcp-topic /ur5/tcp_pose \
  --teleop-joint-cmd-topic /ur5/servo_joint \
  --teleop-gripper-cmd-topic /ur5/gripper_cmd \
  --out-dir /home/lcw/ur5_lerobot/data \
  --task-name task_id \
  --hz 30 \
  --no-depth
```

- `s`: 녹화 시작 (새 episode 생성)
- `q`: 녹화 정지
- `ESC`: 종료

저장 구조:
- `/home/lcw/ur5_lerobot/data/ros/<run_timestamp>/episode_000/*.pkl`
- `/home/lcw/ur5_lerobot/data/ros/<run_timestamp>/episode_001/*.pkl`

카메라 키는 name 기반으로 저장됩니다:
- 예: `wrist_rgb`, `front_center_rgb`, `front_left30_rgb`

## 설치

```bash
cd /home/lcw/ur5_lerobot
pip install -e .
```

## 트러블슈팅
- 카메라 프레임 미도착/타임아웃:
  - realsense-viewer/republish 프로세스 종료 후 재시도
  - 메인보드 USB 포트 직결, 케이블 교체/짧은 케이블 사용
  - 낮은 모드에서 단일 확인: `--width 424 --height 240 --fps 15` 또는 `--fps 6`
- MJPEG 미지원:
  - 일부 장비/펌웨어에서 MJPEG가 광고되지 않습니다. 현재 기본은 YUYV→RGB8입니다.

## 자주 쓰는 /ur5/cmd

브릿지 실행 후 다른 터미널에서:

```bash
# 현재 TCP / Joint 확인
ros2 topic pub --once /ur5/cmd std_msgs/msg/String "{data: 'where'}"

# 현재 Joint를 이름으로 저장
ros2 topic pub --once /ur5/cmd std_msgs/msg/String "{data: 'save observe'}"

# 저장된 포즈 목록
ros2 topic pub --once /ur5/cmd std_msgs/msg/String "{data: 'list'}"

# 저장 포즈로 이동
ros2 topic pub --once /ur5/cmd std_msgs/msg/String "{data: 'go observe'}"
```

## 데이터 변환 / 학습

상세 파이프라인은 기존 스크립트를 사용하세요.
- LeRobot 변환: `scripts/data/convert_to_lerobot.py`
- 학습/실행: `ur5.policies.runner` 및 각 모델 레포 설정

## 프로젝트 구조

자세한 구조는 `PROJECT_STRUCTURE.md` 참고.
