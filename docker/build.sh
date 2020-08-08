#!/bin/bash

# build all docker images in the correct order

cuda/build.sh
ros/build.sh
tensorflow/build.sh
torch/build.sh
transformers/build.sh
dialogpt/build.sh
