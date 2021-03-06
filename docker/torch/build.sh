#!/bin/bash

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

source "$DIRNAME/../config.sh"

docker build \
  -t "${BASE_NAME}/$NAME:${TORCH_VERSION}-${OS}" -t "${BASE_NAME}/$NAME:latest" \
  --build-arg TORCH_VERSION=$TORCH_VERSION \
  --build-arg TORCHVISION_VERSION=$TORCHVISION_VERSION \
  --build-arg TORCHAUDIO_VERSION=$TORCHAUDIO_VERSION \
  --build-arg CUNNN=$CUNNN \
  $1 $DIRNAME
