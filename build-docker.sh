#!/bin/bash

# StreamVGGT Docker Build Script
set -e

echo "🚀 Building StreamVGGT Docker container with GPU support..."

# Check if NVIDIA Docker is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ckpt data outputs

# Build the Docker image
echo "🔨 Building Docker image..."
docker compose build

# Test NVIDIA Docker runtime
echo "🧪 Testing NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Docker runtime is working!"
else
    echo "❌ NVIDIA Docker runtime test failed. Please install nvidia-container-toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "🎉 Build complete! You can now run:"
echo "   docker compose up          # Start the demo"
echo "   docker compose up -d       # Start in background"
echo "   docker compose run --rm streamvggt bash  # Interactive shell"

echo ""
echo "📊 Container will be available at: http://localhost:7860"
echo "💾 Checkpoints will be stored in: ./ckpt/"
echo "📁 Data should be placed in: ./data/"
echo "📤 Outputs will be saved to: ./outputs/"