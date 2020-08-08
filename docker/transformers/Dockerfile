# Hugging Face transformers with pytorch
FROM bdbd/torch:latest as base

SHELL ["/bin/bash", "-c"]
ARG TRANSFORMERS_VERSION=3.0.2
ARG TRANSFORMERS_CACHE=/opt/bdbd_docker/.transformerscache
ENV TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE

RUN pip install transformers==${TRANSFORMERS_VERSION}
