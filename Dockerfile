FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# prefer wheels; avoid compiling
ENV PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

# system deps incl. build tools (covers pycares or any C wheel fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget \
    build-essential pkg-config \
    libgl1 libglib2.0-0 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# install python deps first (cache-friendly)
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# clone kohya-ss where handler expects it (no pip install here)
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git sd-scripts

# app code
WORKDIR /app
COPY . .

# health server + handler
EXPOSE 8000
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
