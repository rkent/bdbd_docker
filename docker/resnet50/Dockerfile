FROM bdbd/tensorflow:latest as base

USER 1000:1000
COPY load_model.py .
RUN python load_model.py
