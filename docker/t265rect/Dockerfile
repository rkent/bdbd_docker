FROM bdbd/ros:latest

# additional dependencies of object_detect
RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-geometry
RUN pip install opencv-python
