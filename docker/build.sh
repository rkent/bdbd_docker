#!/bin/bash

source "config.sh"

# Build the docker images
#docker build -t "${IMAGE_NAME}/cuda:${CUDA_VERSION}-base-${OS}" "cuda/dist/${OS}/${CUDA_VERSION}/base"
#docker build -t "${IMAGE_NAME}/ros:${ROS_VERSION}-${OS}" "ros"
#docker build -t "${IMAGE_NAME}/tf_ros:${TF_VERSION}-${ROS_VERSION}-${OS}" "tensorflow"
#docker build -t "${IMAGE_NAME}/torch:${TORCH_VERSION}-${OS}" "torch"
#docker build -t "${IMAGE_NAME}/transformers:${TRANSFORMERS_VERSION}-${OS}" "transformers"
#docker build -t "${IMAGE_NAME}/deepspeech:${DEEPSPEECH_VERSION}-${OS}" \
#  --build-arg TF_VERSION=${TF1_VERSION} \
#  --build-arg DEEPSPEECH_VERSION=${DEEPSPEECH_VERSION} \
#  --build-arg BASE=${IMAGE_NAME}/ros:${ROS_VERSION}-${OS} \
#  "deepspeech"
docker build -t "${IMAGE_NAME}/hello:${ROS_VERSION}-${OS}" "hello"