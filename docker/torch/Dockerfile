FROM bdbd/tensorflow:latest as base

ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUNNN
RUN pip install \
    torch==${TORCH_VERSION}+${CUNNN} \
    torchvision==${TORCHVISION_VERSION}+${CUNNN} \
    torchaudio==${TORCHAUDIO_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html
