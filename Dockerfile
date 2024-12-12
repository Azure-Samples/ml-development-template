# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

ENV ENV_DIR /ml-core-template-env
WORKDIR $ENV_DIR

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    build-essential \
    gcc \
    g++ \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get install libgomp1 -y


RUN pip install poetry
COPY poetry.lock pyproject.toml $ENV_DIR/

# Rename the file to pyproject.txt
RUN cp $ENV_DIR/pyproject.toml $ENV_DIR/pyproject.txt

RUN poetry config virtualenvs.create false
RUN poetry install