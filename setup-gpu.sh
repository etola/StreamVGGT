#!/bin/bash

# StreamVGGT GPU Setup Script
# Installs NVIDIA Container Toolkit for Docker GPU support

set -e

echo "🚀 Setting up NVIDIA Container Toolkit for StreamVGGT..."
echo ""

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  This script should not be run as root. Please run as regular user with sudo access."
    exit 1
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not detected. Please install NVIDIA drivers first:"
    echo "   sudo apt update && sudo apt install nvidia-driver-545"
    echo "   (or appropriate driver version for your GPU)"
    echo ""
    echo "After installing drivers, reboot and run this script again."
    exit 1
fi

echo "✅ NVIDIA drivers detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed. Please log out and back in for group changes to take effect."
    echo "Then run this script again."
    exit 0
fi

echo "✅ Docker detected"

# Add NVIDIA Container Toolkit repository
echo "📦 Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list and install
echo "⬇️  Installing NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
echo "⚙️  Configuring Docker for NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service
echo "🔄 Restarting Docker service..."
sudo systemctl restart docker

# Test GPU access
echo "🧪 Testing GPU access in Docker..."
if docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU access in Docker is working!"
    echo ""
    echo "🎉 Setup complete! You can now run:"
    echo "   ./build-docker.sh"
    echo ""
else
    echo "❌ GPU test failed. Please check the installation:"
    echo "   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi"
    exit 1
fi