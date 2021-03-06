#!/bin/bash

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

source "$DIRNAME/../config.sh"

docker build \
  -t "${BASE_NAME}/$NAME:${ROS_VERSION}-${OS}" -t "${BASE_NAME}/$NAME:latest" \
  --build-arg ROS_VERSION=$ROS_VERSION $1 \
  --build-arg CUDA_VERSION=$CUDA_VERSION \
  --build-arg OS=$OS \
  $DIRNAME
