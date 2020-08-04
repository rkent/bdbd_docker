#!/bin/bash

source "../config.sh"

docker build -t "${IMAGE_NAME}/hello:${ROS_VERSION}-${OS}" -t ${IMAGE_NAME}/hello:latest .
