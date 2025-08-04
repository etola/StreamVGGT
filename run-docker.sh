#!/bin/bash

# StreamVGGT Docker Run Script
set -e

echo "🚀 Starting StreamVGGT Docker container..."

# Check if image exists
if ! docker image inspect streamvggt:latest > /dev/null 2>&1; then
    echo "📦 Docker image not found. Building first..."
    ./build-docker.sh
fi

# Create directories if they don't exist
mkdir -p ckpt data outputs

# Parse command line arguments
case "${1:-demo}" in
    "demo")
        echo "🎭 Starting demo server..."
        docker compose up
        ;;
    "demo-bg")
        echo "🎭 Starting demo server in background..."
        docker compose up -d
        echo "📊 Demo available at: http://localhost:7860"
        echo "📋 View logs with: docker compose logs -f"
        echo "🛑 Stop with: docker compose down"
        ;;
    "bash")
        echo "💻 Starting interactive bash session..."
        docker compose run --rm streamvggt bash
        ;;
    "train")
        echo "🏋️  Starting training..."
        docker compose run --rm streamvggt bash -c "cd src && python train.py --config-name train"
        ;;
    "eval")
        echo "📊 Starting evaluation..."
        echo "Available evaluation scripts:"
        echo "  - monodepth: bash eval/monodepth/run.sh"
        echo "  - video_depth: bash eval/video_depth/run.sh"
        echo "  - mv_recon: bash eval/mv_recon/run.sh"
        read -p "Enter evaluation type (monodepth/video_depth/mv_recon): " eval_type
        docker compose run --rm streamvggt bash -c "cd src && bash eval/${eval_type}/run.sh"
        ;;
    "gpu-test")
        echo "🧪 Testing GPU access..."
        docker compose run --rm streamvggt python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected')
"
        ;;
    "logs")
        echo "📋 Showing container logs..."
        docker compose logs -f
        ;;
    "stop")
        echo "🛑 Stopping containers..."
        docker compose down
        ;;
    "clean")
        echo "🧹 Cleaning up containers and images..."
        docker compose down
        docker image rm streamvggt:latest 2>/dev/null || true
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  demo      - Start demo server (default)"
        echo "  demo-bg   - Start demo server in background"
        echo "  bash      - Interactive bash session"
        echo "  train     - Run training"
        echo "  eval      - Run evaluation"
        echo "  gpu-test  - Test GPU access"
        echo "  logs      - Show container logs"
        echo "  stop      - Stop containers"
        echo "  clean     - Clean up containers and images"
        ;;
esac