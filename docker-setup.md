# StreamVGGT Docker Setup (GPU-Accelerated)

This Docker setup provides a GPU-accelerated containerized environment for StreamVGGT, solving CUDA compatibility issues and providing optimal performance.

## Prerequisites

1. **NVIDIA GPU** with compatible drivers
2. **Docker** installed  
3. **NVIDIA Container Toolkit** for Docker GPU support

## Easy Setup

### Option 1: Automatic GPU Setup (Recommended)

```bash
# Run the automated setup script
chmod +x setup-gpu.sh
./setup-gpu.sh

# This will install NVIDIA Container Toolkit and configure Docker
```

### Option 2: Manual NVIDIA Container Toolkit Installation

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/nvidia-container-toolkit/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/nvidia-container-toolkit/stable/ubuntu22.04/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

### Build and Run

```bash
# Build the GPU-accelerated container
./build-docker.sh

# Start the demo
docker compose up
```

### Alternative Commands

```bash
# Start in background
docker compose up -d

# Interactive shell with GPU access  
docker compose run --rm streamvggt bash

# Stop the demo
docker compose down
```

## Container Features

- **Base**: Ubuntu 22.04 with CUDA 12.9.1 runtime
- **Python**: 3.11 with optimized packages
- **PyTorch**: 2.4.1 with CUDA 12.4 support (compatible with CUDA 12.9)
- **OpenMP**: System-level libomp-dev for parallel processing
- **GPU Access**: Automatic detection of all available GPUs
- **Volumes**: Persistent storage for checkpoints, data, and outputs

## Usage Examples

### Demo Server
```bash
# Start the Gradio demo (accessible at http://localhost:7860)
docker compose up

# Start in background
docker compose up -d
docker compose logs -f  # View logs
```

### Interactive Development
```bash
# Get a bash shell in the container
docker compose run --rm streamvggt bash

# Test GPU access
docker compose run --rm streamvggt python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

### Training and Evaluation
```bash
# Run training
docker compose run --rm streamvggt bash -c "cd src && python train.py --config-name train"

# Run evaluation
docker compose run --rm streamvggt bash -c "cd src && python eval/monodepth/launch.py"
```

## Directory Structure

The container mounts the following directories:

- `./ckpt/` → `/workspace/StreamVGGT/ckpt/` (model checkpoints)
- `./data/` → `/workspace/StreamVGGT/data/` (datasets)  
- `./outputs/` → `/workspace/StreamVGGT/outputs/` (results)

## Verification

### Check NVIDIA Runtime Installation
```bash
# Check NVIDIA runtime installation
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

### Test StreamVGGT Container
```bash
# Use the convenience script
./run-docker.sh gpu-test

# Or run directly
docker compose run --rm streamvggt python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## Troubleshooting

### NVIDIA Runtime Issues
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get remove nvidia-container-toolkit
sudo apt-get autoremove
./setup-gpu.sh

# Check Docker daemon configuration
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### CUDA Version Mismatch
The container uses CUDA 12.9.1 runtime with PyTorch compiled for CUDA 12.4, which is forward-compatible. Your host CUDA drivers should be >= 12.4.

### Memory Issues
```bash
# Increase shared memory if needed
docker compose run --rm --shm-size=16gb streamvggt bash
```

## Environment Variables

The container sets the following GPU-related environment variables:
- `NVIDIA_VISIBLE_DEVICES=all` (use all GPUs)
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (enable compute and utility)

## Performance Tips

1. **Multiple GPUs**: The container automatically detects and can use all available GPUs
2. **Memory**: 8GB shared memory is allocated by default for large models
3. **Storage**: Use SSD storage for `./data/` directory for faster I/O
4. **Monitoring**: Use `nvidia-smi` inside the container to monitor GPU usage

## Scripts Reference

- `build-docker.sh` - Build the GPU-accelerated container
- `run-docker.sh` - Convenient wrapper for common operations
- `setup-gpu.sh` - Install NVIDIA Container Toolkit
- `docker-compose.yml` - Container orchestration configuration