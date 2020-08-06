FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3

WORKDIR /opt/app

COPY test.py .
RUN chown 1000:1000 *

#CMD sh -c "while true; do $(echo date); sleep 1; done"
CMD python3 test.py