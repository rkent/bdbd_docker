# I can get ROS nodes to work in docker if started from bash, but not from
# the docker python API. This is an experiment to see if I run a subprocess
# from python, will that start the node and work.

import subprocess
subprocess.run(['/opt/bdbd_docker/docker/hello/run.sh'])
