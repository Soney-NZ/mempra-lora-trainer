FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Pre-install Kohya training scripts in the container
WORKDIR /tmp
RUN git clone https://github.com/bmaltais/kohya_ss.git sd-scripts && \
    cd sd-scripts && \
    sed -i '/^-e \.\//d; /^file:\/\/\/.*sd-scripts$/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Install our application dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Start RunPod serverless handler
CMD ["python", "-u", "handler.py"]