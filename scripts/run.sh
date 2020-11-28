#!/usr/bin/env bash

IMAGE_URI="mikomel/machine-number-sense:latest"

docker run --rm \
  --mount type=volume,source=datasets,destination=/app/datasets \
  --mount type=volume,source=logs,destination=/app/logs \
  --gpus all \
  --ipc host \
  "$IMAGE_URI"
