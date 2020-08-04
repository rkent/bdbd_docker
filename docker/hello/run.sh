#!/bin/bash

# launch the hello docker container

docker run \
  -u $(id -u):$(id -g) \
  -d \
  --hostname=hello \
  --mount type=bind,source="/opt/bdbd_docker",target=/opt/bdbd_docker \
  --network 'host' \
  -e "ROS_MASTER_URI=http://nano.dryrain.org:11311/" \
  -e "ROS_IP=192.168.0.59" \
  --name=hello \
  --rm \
  bdbd/hello:latest
