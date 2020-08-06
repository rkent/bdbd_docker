# Hugging Face transformers with pytorch
ARG OS=ubuntu20.04
FROM bdbd/torch:latest as base

SHELL ["/bin/bash", "-c"]
ARG TRANSFORMERS_VERSION=3.0.2
ARG TRANSFORMERS_CACHE=/opt/bdbd_docker/.transformerscache
ARG TORCH_VERSION="1.6.0"
ENV TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE

RUN pip install torch==${TORCH_VERSION}+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers==${TRANSFORMERS_VERSION}

COPY test.py /opt/app