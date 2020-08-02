#!/bin/bash

# just a demo of how to run a file

docker run \
  -u $(id -u):$(id -g) \
  -it \
  --gpus all \
  --hostname=transformers \
  --mount type=bind,source="/home/kent/.cache/torch/transformers",target=/var/cache/transformers \
  --mount type=bind,source="/home/kent/github/rkent/bdbd",target=/opt/bdbd \
  bdbd/transformers:3.0.2-ubuntu20.04 \
  /bin/bash
