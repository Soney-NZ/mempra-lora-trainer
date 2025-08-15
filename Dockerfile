# Production Dockerfile for Flux LoRA Training - PyTorch Fixed
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Build argument for cache busting
ARG BUILD_ID=12

# Environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/transformers \
    WORKSPACE_SYNC=false \
    AUTO_UPDATE=false \
    PORT_HEALTH=8000

# System dependencies with proper cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl unzip \
    build-essential pkg-config \
    libglib2.0-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace directory
WORKDIR /workspace

# PyTorch compatibility fix - Remove base image versions and install fixed versions
RUN pip uninstall -y torch torchvision torchaudio || true

# Install PyTorch with CUDA 12.4 wheels (from ComfyUI notebooks)
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch compatibility - Test the problematic operation that was failing
RUN python -c "import torch, torchvision; from torchvision import _meta_registrations; print(f'✅ PyTorch {torch.__version__} and torchvision {torchvision.__version__} compatible')"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Clone and setup Kohya SS training scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git && \
    cd sd-scripts && \
    git checkout main && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories (no persistent cache)
RUN mkdir -p /tmp/model_cache \
    /workspace/temp \
    /tmp \
    && chmod 755 /tmp/model_cache

# Copy all production files
COPY handler-production.py /workspace/handler.py
COPY enhanced-comfyui-config.py /workspace/
COPY healthcheck.py /workspace/
COPY start.sh /workspace/

# Make start script executable
RUN chmod +x /workspace/start.sh

# RunPod health contract - Use /ping endpoint as per documentation
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Expose the health port that RunPod expects
ENV PORT_HEALTH=8000
EXPOSE 8000

# Use start script to manage health server + job loop properly
CMD ["/workspace/start.sh"]