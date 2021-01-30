#!/bin/bash
source /opt/bdbd_docker/devel/setup.bash
/opt/bdbd_docker/src/bdbd_docker/nodes/$ROSNODE &
NODEPID=$!

# setup to send signals to node to terminate cleanly
trap "kill -s SIGTERM $NODEPID" SIGTERM

# loop to keep alive
wait $NODEPID

# we seem to need a little time for the ros cleanup to complete
sleep 1
