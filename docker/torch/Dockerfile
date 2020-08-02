ARG UBUNTU_VERSION=20.04
ARG CUDA=10.1
FROM bdbd/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
