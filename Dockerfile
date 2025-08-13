# RunPod LoRA Training Container
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    diffusers==0.24.0 \
    transformers==4.36.0 \
    accelerate==0.24.1 \
    bitsandbytes==0.41.3 \
    xformers==0.0.22.post7 \
    wandb \
    opencv-python \
    pillow \
    numpy \
    torch \
    torchvision \
    peft \
    safetensors \
    requests \
    cloudinary

# Clone Kohya training scripts
WORKDIR /workspace
RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR /workspace/sd-scripts
RUN pip install -r requirements.txt

# Copy training script
COPY train_lora.py /workspace/train_lora.py
COPY handler.py /workspace/handler.py

# Set working directory
WORKDIR /workspace

# Set the handler
CMD ["python", "handler.py"]