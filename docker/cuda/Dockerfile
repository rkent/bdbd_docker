FROM ubuntu:20.04
LABEL maintainer "R. Kent James <kent@caspia.com>"
LABEL description "nvidia base adapted for Ubuntu 20.04"
ARG CUDA_VERSION
ENV CUDA_VERSION=${CUDA_VERSION}

# Stuff added by RKJ we'll always want
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    nano \
    iproute2 \
    git \
    iputils-ping \
    gnupg2 curl ca-certificates

# Some tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata keyboard-configuration

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN useradd -ms /bin/bash bdbd

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-${CUDA_VERSION/./-} \
    cuda-${CUDA_VERSION/./-} \
    cuda-cudart-${CUDA_VERSION/./-} \
    cuda-compat-${CUDA_VERSION/./-}

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"
