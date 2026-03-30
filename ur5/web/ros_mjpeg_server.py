# Thin wrapper to keep public path as `ur5.web.ros_mjpeg_server`
from ur5.rfm.web.ros_mjpeg_server import *  # re-export
from ur5.rfm.web.ros_mjpeg_server import main as main

if __name__ == "__main__":
    main()

