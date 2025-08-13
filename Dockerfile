FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libgl1 libglib2.0-0 wget && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /tmp
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git sd-scripts && \
    pip install --no-cache-dir -r /tmp/sd-scripts/requirements.txt

WORKDIR /app
COPY . .

EXPOSE 8000

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
