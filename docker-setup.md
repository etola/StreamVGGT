# StreamVGGT Docker Setup (GPU)

This Docker setup provides a containerized environment for StreamVGGT with GPU support, solving CUDA compatibility issues.

## Prerequisites

1. **NVIDIA Docker Runtime**:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Docker Compose** (if not already installed):
   ```bash
   sudo apt-get install docker-compose-plugin
   ```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the container
docker compose up --build

# Or run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the container
docker compose down
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t streamvggt:latest .

# Run the container
docker run --gpus all \
  -p 7860:7860 \
  -v $(pwd)/ckpt:/workspace/StreamVGGT/ckpt \
  -v $(pwd)/data:/workspace/StreamVGGT/data \
  -v $(pwd)/outputs:/workspace/StreamVGGT/outputs \
  --shm-size=8g \
  streamvggt:latest
```

## Container Features

- **Base**: Ubuntu 22.04 with CUDA 12.9.1 runtime
- **Python**: 3.11 (as specified in original setup)
- **PyTorch**: 2.3.1 with CUDA 12.4 support (compatible with CUDA 12.9)
- **OpenMP**: System-level `libomp-dev` (equivalent to `conda install llvm-openmp<16`)
- **Port**: 7860 (Gradio default)
- **Health Check**: Verifies CUDA availability

## Directory Structure

```
StreamVGGT/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── docker-setup.md
├── ckpt/              # Model checkpoints (mounted volume)
├── data/              # Datasets (mounted volume)
├── outputs/           # Results (mounted volume)
└── ...
```

## Mounted Volumes

The Docker setup uses volumes for persistent storage:
- `./ckpt` → Model checkpoints
- `./data` → Training/evaluation datasets  
- `./outputs` → Generated results and outputs

## Usage Examples

### Running the Demo
```bash
# Start the container with demo
docker compose up

# Access at: http://localhost:7860
```

### Interactive Development
```bash
# Start container with bash
docker compose run --rm streamvggt bash

# Or exec into running container
docker compose exec streamvggt bash
```

### Training/Evaluation
```bash
# Run training
docker compose exec streamvggt bash -c "cd src && python train.py --config-name train"

# Run evaluation
docker compose exec streamvggt bash -c "cd src && bash eval/monodepth/run.sh"
```

## GPU Verification

To verify GPU access in the container:

```bash
docker compose exec streamvggt python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

## Troubleshooting

### NVIDIA Runtime Issues
```bash
# Check NVIDIA runtime installation
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi

# If above fails, reinstall nvidia-container-toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Memory Issues
- Increase `shm_size` in docker-compose.yml if needed
- Monitor GPU memory: `nvidia-smi`

### Port Conflicts
```bash
# Change port mapping in docker-compose.yml
ports:
  - "8080:7860"  # Use port 8080 instead
```

## Performance Notes

- **Build time**: ~10-15 minutes (depending on internet speed)
- **Image size**: ~8-10 GB
- **GPU memory**: Depends on model size and batch size
- **Shared memory**: 8GB allocated (adjust if needed)

## Comparison with Virtual Environment

| Method | Pros | Cons |
|--------|------|------|
| **Virtual Env** | Lighter, faster setup | CUDA compatibility issues |
| **Docker** | Isolated, reproducible | Larger, more complex |

The Docker approach provides better isolation and avoids the CUDA compatibility issues you encountered with the virtual environment setup.