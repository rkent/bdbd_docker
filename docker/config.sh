# This file contain configuration variables for the docker containers

# deprecated
export IMAGE_NAME="bdbd" 

export BASE_NAME="bdbd"
export CUDA_VERSION="11.0"
export CUDA_MAJOR_VERSION=${CUDA_VERSION%.*}
export ORB_SLAM3_VERSION="v0.3-beta"
export OS="ubuntu20.04"
export TF_VERSION="2.4.0"
export ROS_VERSION="noetic"
export CUNNN="cu110"
export TORCH_VERSION="1.7.1"
export TORCHVISION_VERSION="0.8.2"
export TORCHAUDIO_VERSION="0.7.2"
export TRANSFORMERS_VERSION="4.2.1"
export DEEPSPEECH_VERSION="0.9.3"
export TF1_VERSION="1.15.0"
export ROS_MASTER_URI=http://nano.dryrain.org:11311/
export ROS_IP="192.168.0.59"
export ROS_NAMESPACE="/bdbd"
export TRANSFORMERS_CACHE=/opt/bdbd_docker/.transformerscache
export TF_MODELS_VERSION="2.4.0"
# git version as of 2021-02-06
export TF_MODELS_GIT="99b8390c3ab66aa1753ab7c606bda0a0d80e455a"
