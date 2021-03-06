#!/bin/bash

source "../config.sh"
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

docker build -t "${IMAGE_NAME}/$NAME:${ROS_VERSION}-${OS}" -t ${IMAGE_NAME}/$NAME:latest .
