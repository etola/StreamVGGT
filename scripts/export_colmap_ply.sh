#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/export_colmap_ply.sh <target_dir> [--out_dir <output_dir>]
# <target_dir> should be relative to the project root and contain an images/ subfolder.

if [ $# -lt 1 ]; then
  echo "Usage: $0 <target_dir> [--out_dir <output_dir>]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../" && pwd)"
TARGET_DIR="$1"

HOST_DIR=$(realpath "$TARGET_DIR")
if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi
echo "Running StreamVGGT container with directory: $HOST_DIR"

# Pass any extra args (e.g. --out_dir ...) to the python script
shift
EXTRA_ARGS="$@"

# Check GPU compatibility
echo "🔍 Checking GPU compatibility..."
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "No GPU")
echo "GPU detected: $GPU_INFO"

# Special handling for RTX 5080
if [[ "$GPU_INFO" == *"RTX 5080"* ]]; then
    echo "⚠️ RTX 5080 detected - this GPU has sm_120 capability"
    echo "💡 Current PyTorch version may not support this GPU"
    echo "🔄 Script will automatically fallback to CPU if needed"
fi

DOCKER_ARGS=(
  -it --rm
  --gpus all
  -v "$ROOT_DIR":/workspace/StreamVGGT 
  -v "$HOST_DIR":/working
  --workdir /workspace/StreamVGGT
  -e MPLCONFIGDIR=/tmp/matplotlib
  --user $(id -u):$(id -g)
  -e CUDA_LAUNCH_BLOCKING=1
)

# Test GPU compatibility first
echo "🧪 Testing GPU compatibility..."
if docker run --rm --runtime=nvidia streamvggt nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU runtime available"
    
    # Test PyTorch CUDA compatibility
    if docker run --rm --gpus all streamvggt python -c "
import torch
try:
    if torch.cuda.is_available():
        test = torch.tensor([1.0]).cuda()
        result = test + 1
        print('✅ CUDA compatible')
    else:
        print('❌ CUDA not available')
        exit(1)
except Exception as e:
    if 'no kernel image' in str(e) or 'sm_120' in str(e):
        print('❌ CUDA compatibility issue (RTX 5080 sm_120)')
        exit(2)
    else:
        raise
" 2>/dev/null; then
        echo "✅ PyTorch CUDA compatible - using GPU"
        USE_GPU="true"
    else
        echo "⚠️ PyTorch CUDA compatibility issue - will use CPU fallback"
        USE_GPU="false"
    fi
else
    echo "❌ GPU runtime not available - using CPU"
    USE_GPU="false"
fi

# Add environment variable to force CPU if needed
if [[ "$USE_GPU" == "false" ]]; then
    DOCKER_ARGS+=(-e CUDA_VISIBLE_DEVICES="")
    echo "🔄 Forcing CPU execution"
fi

echo "🚀 Starting export process..."

# Use /working as the target directory since we mounted the host directory there
docker run "${DOCKER_ARGS[@]}" streamvggt \
  python tools/export_colmap_ply.py $EXTRA_ARGS

