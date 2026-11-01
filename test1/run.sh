#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t rocmpytorch:test "$SCRIPT_DIR"

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16g \
  -v "$SCRIPT_DIR/docker_data":/app \
  -w /app \
  rocmpytorch:test bash
