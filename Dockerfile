FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG BUILD_ID=2  # bump this number to force Docker to rebuild this layer and everything after

ENV PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

# System deps + full build toolchain + Python headers + SSL/ffi for any C wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget \
    build-essential pkg-config \
    python3-dev \
    libssl-dev libffi-dev \
    libgl1 libglib2.0-0 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Keep pip under 25 for stability; use verbose logs for diagnosis if anything fails
RUN python -m pip install --upgrade "pip<25" setuptools wheel && \
    python -m pip --version && \
    python -m pip install -vvv -r /app/requirements.txt

# ---------- sd-scripts (no pip install of its requirements) ----------
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git sd-scripts

# ---------- App code ----------
WORKDIR /app
COPY . .

EXPOSE 8000
RUN chmod +x /start.sh
CMD ["/start.sh"]
