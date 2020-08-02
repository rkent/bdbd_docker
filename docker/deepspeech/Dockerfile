ARG BASE="bdbd/ros:noetic-ubuntu20.04"
FROM ${BASE}

ARG TF_VERSION
ARG DEEPSPEECH_VERSION
RUN apt-get update && apt-get install -y software-properties-common curl nano iproute2
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7

RUN ln -s -f /usr/bin/python3.7 /usr/bin/python3
RUN ln -s -f /usr/bin/python3.7 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install tensorflow==${TF_VERSION}
RUN pip install deepspeech==${DEEPSPEECH_VERSION}

WORKDIR /opt/deepspeech
RUN curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v${DEEPSPEECH_VERSION}/deepspeech-${DEEPSPEECH_VERSION}-models.pbmm
RUN curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v${DEEPSPEECH_VERSION}/deepspeech-${DEEPSPEECH_VERSION}-models.scorer
RUN curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v${DEEPSPEECH_VERSION}/audio-${DEEPSPEECH_VERSION}.tar.gz
RUN tar xvf audio-${DEEPSPEECH_VERSION}.tar.gz

