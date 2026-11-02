#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build -t rocmpytorch:test4 "$SCRIPT_DIR"

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16g \
  -v "$SCRIPT_DIR/docker_data":/app \
  -v "$REPO_DIR/data":/app/data \
  -w /app \
  rocmpytorch:test4 bash

