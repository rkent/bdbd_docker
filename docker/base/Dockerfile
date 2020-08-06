FROM ubuntu:20.04

# Stuff added by RKJ we'll always want
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    nano \
    iproute2 \
    git \
    iputils-ping

# Some tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN useradd -ms /bin/bash bdbd
