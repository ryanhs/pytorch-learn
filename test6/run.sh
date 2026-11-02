#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build -t food-label-pilot:test6 "$SCRIPT_DIR"

docker run --rm -it \
  --network=host \
  -v "$SCRIPT_DIR/docker_data":/app \
  -v "$SCRIPT_DIR/data":/app/data \
  -w /app \
  food-label-pilot:test6 \
  bash
