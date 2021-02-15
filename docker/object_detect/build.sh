#!/bin/bash

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

source "$DIRNAME/../config.sh"

docker build \
  -t "${BASE_NAME}/$NAME:${TF_MODELS_VERSION}-${OS}" -t "${BASE_NAME}/$NAME:latest" \
  --build-arg TF_MODELS_VERSION=$TF_MODELS_VERSION \
  --build-arg TF_MODELS_GIT=$TF_MODELS_GIT \
  $1 $DIRNAME
