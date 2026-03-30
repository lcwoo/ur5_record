# Thin wrapper to keep public path as `ur5.cameras.realsense_ros_multi_publisher`
from ur5.rfm.cameras.realsense_ros_multi_publisher import *  # re-export symbols
from ur5.rfm.cameras.realsense_ros_multi_publisher import main as main  # entry point

if __name__ == "__main__":
    main()

