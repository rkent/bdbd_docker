FROM bdbd/tensorflow:latest

ARG TF_MODELS_VERSION
ARG TF_MODELS_GIT

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

WORKDIR /home/bdbd

# download models release
RUN wget "https://github.com/tensorflow/models/archive/${TF_MODELS_GIT}.zip"
RUN unzip "${TF_MODELS_GIT}.zip" && rm "${TF_MODELS_GIT}.zip"
#RUN ln -s "models-${TF_MODELS_GIT}" models 
RUN mv "models-${TF_MODELS_GIT}" models

# Compile protobuf configs
WORKDIR /home/bdbd/models/research/
RUN protoc object_detection/protos/*.proto --python_out=.

WORKDIR /home/bdbd/models/research/
RUN cp object_detection/packages/tf2/setup.py ./
RUN python -m pip install .
ENV PATH="/home/bdbd/.local/bin:${PATH}"

# additional dependencies of object_detect
RUN apt-get install -y \
    ros-noetic-cv-bridge
ENV TF_CPP_MIN_LOG_LEVEL 3

# should not be needed, but testing
#CMD ["/opt/app/rosrun.sh"]
