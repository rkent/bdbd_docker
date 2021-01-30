#!/bin/bash

# Runs a docker container using the following name convention: same name is used for:
# * basename of directory containing this scrpt
# * hostname
# * container name
# * image name (with /bdbd/ prefix)
# * environment variable ROSNODE started by the container

source "../config.sh"

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)
echo "starting DOCKER container for $NAME"

# launch the hello docker container
docker container rm --force $NAME > /dev/null 2>&1

docker run \
  -u $(id -u):$(id -g) \
  -d \
  --hostname=$NAME \
  --mount type=bind,source="/opt/bdbd_docker",target=/opt/bdbd_docker \
  --network 'host' \
  -e "ROS_MASTER_URI=$ROS_MASTER_URI" \
  -e "ROS_IP=$ROS_IP" \
  -e "ROSNODE=$NAME" \
  -e "ROS_NAMESPACE=$ROS_NAMESPACE" \
  --name=$NAME \
  $1 \
  bdbd/$NAME:latest $2
