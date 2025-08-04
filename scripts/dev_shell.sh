#!/usr/bin/env bash
set -euo pipefail

# Usage: ./docker/dev_shell.sh /path/to/data

DATA_DIR="${1:-$HOME/data}"
ROOT_DIR="$(cd "$(dirname "$0")/../" && pwd)"

# Create data dir if it doesn't exist
mkdir -p "$DATA_DIR"

echo "[INFO] Launching interactive shell in Docker..."
echo "       Project root: $ROOT_DIR"
echo "       Data dir:    $DATA_DIR"

docker run --rm -it --gpus all \
  -v "$ROOT_DIR":/workspace/StreamVGGT \
  -v "$DATA_DIR":/data \
  --workdir /workspace/StreamVGGT \
  streamvggt:latest \
  bash
