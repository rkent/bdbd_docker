#!/bin/bash

#https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NAME=$(basename $DIRNAME)

source "$DIRNAME/../config.sh"

docker build \
  -t "${BASE_NAME}/$NAME:latest" $1\
  $DIRNAME
