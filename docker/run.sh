#!/bin/bash

# Runs a docker container using the following name convention: same name is used for:
# * basename of directory containing this scrpt
# * hostname
# * container name
# * image name (with /bdbd/ prefix)
# * environment variable ROSNODE started by the container

# usage: ./run.sh [program] [options]

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)
echo "DIRNAME is $DIRNAME"
echo "starting DOCKER container for $NAME"
echo "options $1 command $2"

source "$DIRNAME/../config.sh"
# launch the hello docker container
docker container rm --force $NAME > /dev/null 2>&1

if [ -z $1 ]; then
    OPTIONS=-d
else
    OPTIONS=$1
fi

docker run \
  -u $(id -u):$(id -g) \
  --hostname=$NAME \
  --mount type=bind,source="/opt/bdbd_docker",target=/opt/bdbd_docker \
  --network 'host' \
  -e "ROS_MASTER_URI=$ROS_MASTER_URI" \
  -e "ROS_IP=$ROS_IP" \
  -e "ROSNODE=$NAME" \
  -e "ROS_NAMESPACE=$ROS_NAMESPACE" \
  --name=$NAME \
  --gpus all \
  $OPTIONS \
  bdbd/$NAME:latest $2
