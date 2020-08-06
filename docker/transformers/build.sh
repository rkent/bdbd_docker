#!/bin/bash

source "../config.sh"
#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

docker build \
  -t "${BASE_NAME}/$NAME:${TRANSFORMERS_VERSION}-${OS}" -t "${BASE_NAME}/$NAME:latest" \
  --build-arg TRANSFORMERS_VERSION=$TRANSFORMERS_VERSION \
  --build-arg OS=$OS \
  --build-arg TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
  --build-arg TORCH_VERSION=$TORCH_VERSION \
  .
