FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG BUILD_ID=5  # bump this number to force Docker to rebuild this layer and everything after

ENV PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps + full build toolchain + Python headers + SSL/ffi for any C wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl \
    build-essential pkg-config \
    python3-dev python3-pip \
    libssl-dev libffi-dev \
    libgl1 libglib2.0-0 \
    libglib2.0-dev \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt

# Update pip to latest stable and install dependencies with better error handling
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip --version && \
    python --version && \
    python -m pip install --no-cache-dir \
      --constraint /app/constraints.txt \
      --requirement /app/requirements.txt

# ---------- sd-scripts (no pip install of its requirements) ----------
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git sd-scripts

# ---------- App code ----------
WORKDIR /app
COPY . .

EXPOSE 8000
RUN chmod +x start.sh
CMD ["./start.sh"]
