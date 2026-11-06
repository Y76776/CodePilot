FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git bash coreutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /testbed
