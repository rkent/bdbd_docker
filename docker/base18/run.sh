PGM_DEFAULT=terminator
PGM=${1:-$PGM_DEFAULT}
docker run -it --rm \
    --name="base18" \
    -h "base18" \
    -e "DISPLAY=$DISPLAY" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    bdbd/base18 $PGM
