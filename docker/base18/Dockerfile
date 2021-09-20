FROM ubuntu:18.04

# Demo of graphical program access (terminator and pcmanfm)

# setup for non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
# Don't require -y in apt-get commands to continue
RUN echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf
RUN apt-get update && apt-get install tzdata keyboard-configuration

RUN apt-get install\
    python3 \
    python3-pip

# Some tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Stuff added by RKJ we'll always want
RUN apt-get install \
    nano \
    iproute2 \
    git \
    iputils-ping \
    sudo \
    terminator \
    pcmanfm \
    dbus \ 
    dbus-x11 \
    unzip \
    curl \
    libcanberra-gtk-module

RUN useradd -ms /bin/bash bdbd

# remove spurious error messages
RUN mkdir -p /root/.config/terminator
RUN mkdir -p /root/Templates
RUN touch /root/.config/terminator/config
ENV NO_AT_BRIDGE=1

