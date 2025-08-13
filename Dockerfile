FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Prefer wheels; keep cache off for leaner image
ENV PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

# System deps incl. build tools (safety net for any C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget \
    build-essential pkg-config \
    libgl1 libglib2.0-0 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ---------- Python deps (pin pip<25, wheels-only) ----------
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Keep pip below 25 to avoid backend quirks; ensure modern build tools
RUN python -m pip install --upgrade "pip<25" setuptools wheel \
 && python -m pip --version \
 && python -m pip install --only-binary=:all: -r /app/requirements.txt

# ---------- sd-scripts (no pip install of its requirements) ----------
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git sd-scripts

# ---------- App code ----------
WORKDIR /app
COPY . .

# Quick sanity import at build time for fast feedback
RUN python - <<'PY'
import torch, transformers, diffusers, bitsandbytes as bnb
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("bitsandbytes OK")
PY

EXPOSE 8000
RUN chmod +x /start.sh
CMD ["/start.sh"]
