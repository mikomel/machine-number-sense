#!/usr/bin/env bash

IMAGE_URI="mikomel/machine-number-sense:latest"

docker build -t "$IMAGE_URI" -f Dockerfile .
