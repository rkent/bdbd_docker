#!/bin/bash

# Runs a docker container using the following name convention: same name is used for:
# * basename of directory containing this scrpt
# * hostname
# * container name
# * image name (with /bdbd/ prefix)
# * environment variable ROSNODE started by the container

source "../config.sh"

# launch the hello docker container
docker container rm --force test > /dev/null 2>&1

docker run \
  --name=test \
  -it \
  bdbd/test:latest
