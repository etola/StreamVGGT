# Use CUDA runtime base image compatible with CUDA 12.x
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/StreamVGGT/src:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    libomp-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /workspace/StreamVGGT

# Copy requirements first for better caching
COPY requirements.txt requirements_demo.txt ./

# Install Python packages directly to system Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt && \
    pip install -r requirements_demo.txt

# Copy the rest of the application
COPY . .

# Create directories for checkpoints and data
RUN mkdir -p ckpt data/eval data/train

# Expose port for Gradio
EXPOSE 7860

# Health check to verify CUDA availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Default command to run the demo
CMD ["python", "demo_gradio.py", "--server_name", "0.0.0.0", "--server_port", "7860"]