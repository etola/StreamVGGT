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

# Test NVIDIA Docker runtime (required for GPU setup)
echo "🧪 Testing NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA Docker runtime is working!"
    echo "🚀 GPU acceleration ready!"
else
    echo "❌ NVIDIA Docker runtime required but not available."
    echo ""
    echo "🔧 To install NVIDIA Container Toolkit:"
    echo "   ./setup-gpu.sh"
    echo ""
    echo "   Or manually:"
    echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "   sudo systemctl restart docker"
    echo ""
    echo "📖 Documentation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "⚠️  After installation, please run this script again."
    exit 1
fi

echo ""
echo "🎉 Build complete! GPU-accelerated StreamVGGT is ready!"
echo ""
echo "🚀 Run the GPU-accelerated demo:"
echo "   docker compose up          # Start the demo"
echo "   docker compose up -d       # Start in background"
echo "   docker compose run --rm streamvggt bash   # Interactive shell"
echo ""
echo "🖥️  Stop the demo:"
echo "   docker compose down        # Stop and remove containers"
echo ""
echo "📊 Demo will be available at: http://localhost:7860"
echo "💾 Checkpoints will be stored in: ./ckpt/"
echo "📁 Data should be placed in: ./data/"
echo "📤 Outputs will be saved to: ./outputs/"
echo ""
echo "💡 Tip: The container will automatically use all available GPUs"