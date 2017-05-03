# YOLO ROS Wrapper 

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

**Author** Ayush Gaud

### Dependencies
1. CUDA  
2. CUDNN (Optional, can be disabled in the darknet Makefile)
2. ROS Kinetic/Indigo


### Instructions

```
cd $ROS_WORKSPACE/src
git clone https://github.com/ayushgaud/darknet_ros.git
git submodule init
git submodule update
cd darknet
make darknet-cpp-shared
cd $ROS_WORKSPACE
catkin_make --pkg darknet_ros
roslaunch darknet_ros object_detect.launch
```
### Contributers

Based on original work of Joseph Redmon https://github.com/prabindh/darknet and its C++ port by Prabindh Sundareson https://github.com/prabindh/darknet.git
