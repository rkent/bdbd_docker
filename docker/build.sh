#!/bin/bash

# build all docker images in the correct order

cuda/build.sh $1
ros/build.sh $1
tensorflow/build.sh $1
torch/build.sh $1
transformers/build.sh $1
dialogpt/build.sh $1
