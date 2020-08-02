ARG BASE="bdbd/cuda:10.1-base-ubuntu20.04"
FROM ${BASE}

ARG ROS_VERSION="noetic"
# Not sure if this is needed, I still get an keyboard-configuration hang with it
COPY keyboard /etc/default/keyboard

RUN apt install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y ros-${ROS_VERSION}-desktop-full
