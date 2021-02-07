ARG CUDA_VERSION
ARG OS
FROM bdbd/cuda:${CUDA_VERSION}-${OS}

ARG ROS_VERSION="noetic"
# Not sure if this is needed, I still get a keyboard-configuration hang with it
COPY keyboard /etc/default/keyboard

RUN apt-get update && apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ros-${ROS_VERSION}-ros-base

#https://stackoverflow.com/questions/29663459/python-app-does-not-print-anything-when-running-detached-in-docker
ENV PYTHONUNBUFFERED=1

RUN mkdir /opt/app && \
    chmod 777 /opt/app

WORKDIR /opt/app
COPY rosrun.sh .
RUN chmod 755 *.* && chown 1000:1000 *.*

CMD ["/opt/app/rosrun.sh"]
