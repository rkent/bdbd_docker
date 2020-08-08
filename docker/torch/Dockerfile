FROM bdbd/tensorflow:latest as base

ARG TORCH_VERSION="1.6.0"
RUN pip install torch==${TORCH_VERSION}+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
