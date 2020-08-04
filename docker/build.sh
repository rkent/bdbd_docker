#!/bin/bash

source "config.sh"

# Build the docker images
#docker build -t "${IMAGE_NAME}/cuda:${CUDA_VERSION}-base-${OS}" -t ${IMAGE_NAME}/cuda:latest "cuda/dist/${OS}/${CUDA_VERSION}/base"
#docker build -t "${IMAGE_NAME}/ros:${ROS_VERSION}-${OS}" -t ${IMAGE_NAME}/ros:latest "ros"
#docker build -t "${IMAGE_NAME}/tf_ros:${TF_VERSION}-${ROS_VERSION}-${OS}" -t ${IMAGE_NAME}/tf_ros:latest"tensorflow"
#docker build -t "${IMAGE_NAME}/torch:${TORCH_VERSION}-${OS}" -t ${IMAGE_NAME}/torch:latest "torch"
#docker build -t "${IMAGE_NAME}/transformers:${TRANSFORMERS_VERSION}-${OS}" -t ${IMAGE_NAME}/transformers:latest "transformers"
#docker build -t "${IMAGE_NAME}/deepspeech:${DEEPSPEECH_VERSION}-${OS}" -t ${IMAGE_NAME}/deepspeech:latest\
#  --build-arg TF_VERSION=${TF1_VERSION} \
#  --build-arg DEEPSPEECH_VERSION=${DEEPSPEECH_VERSION} \
#  --build-arg BASE=${IMAGE_NAME}/ros:${ROS_VERSION}-${OS} \
#  "deepspeech"
docker build -t "${IMAGE_NAME}/hello:${ROS_VERSION}-${OS}" -t ${IMAGE_NAME}/hello:latest "hello"