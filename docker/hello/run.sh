#!/bin/bash

# Runs a docker container using the following name convention: same name is used for:
# * basename of directory containing this scrpt
# * hostname
# * container name
# * image name (with /bdbd/ prefix)
# * environment variable ROSNODE started by the container

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)
echo "starting DOCKER container for $NAME"

# launch the hello docker container

docker run \
  -u $(id -u):$(id -g) \
  -d \
  --hostname=$NAME \
  --mount type=bind,source="/opt/bdbd_docker",target=/opt/bdbd_docker \
  --network 'host' \
  -e "ROS_MASTER_URI=http://nano.dryrain.org:11311/" \
  -e "ROS_IP=192.168.0.59" \
  -e "ROSNODE=$NAME" \
  -e "ROS_NAMESPACE=/bdbd" \
  --name=$NAME \
  --rm \
  bdbd/$NAME:latest
