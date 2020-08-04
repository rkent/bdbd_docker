ARG BASE="bdbd/ros:latest"
FROM ${BASE}

SHELL ["/bin/bash", "-c"]

RUN mkdir /opt/app
RUN chmod 777 /opt/app

WORKDIR /opt/app
COPY rosrun.sh .
RUN chmod 755 rosrun.sh
RUN chown 1000:1000 *.*

CMD "./rosrun.sh"
