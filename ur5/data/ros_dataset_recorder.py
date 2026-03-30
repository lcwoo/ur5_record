# Thin wrapper to keep public path as `ur5.data.ros_dataset_recorder`
from ur5.rfm.data.ros_dataset_recorder import *  # re-export
from ur5.rfm.data.ros_dataset_recorder import main as main

if __name__ == "__main__":
    main()

