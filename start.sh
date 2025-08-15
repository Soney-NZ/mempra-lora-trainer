#!/bin/bash
set -e

echo "[Start] 🚀 Starting Flux LoRA Training Service with RunPod Health Contract"

# Create flag files for health checks
mkdir -p /tmp
touch /tmp/initializing
echo "[Start] 📋 Health flags initialized"

# Start health server in background (non-blocking for /ping endpoint)
echo "[Start] 🏥 Starting health server on port ${PORT_HEALTH:-8000}..."
python /workspace/healthcheck.py &
HEALTH_PID=$!

# Give health server time to start
sleep 3

# Verify health server is running
if ! curl -f http://localhost:${PORT_HEALTH:-8000}/health > /dev/null 2>&1; then
    echo "[Start] ❌ Health server failed to start"
    exit 1
fi

echo "[Start] ✅ Health server running (PID: $HEALTH_PID)"

# Initialize worker components (model cache, etc.) while health returns 204/503
echo "[Start] 🔄 Initializing worker components..."

# Run initialization in background so we can monitor it
python -c "
import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('init')

logger.info('[Init] Starting model cache and dependencies...')

try:
    # Step 1: Verify Python environment
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    logger.info(f'[Init] ✅ PyTorch {torch.__version__} loaded')
    
    # Step 2: Create cache directories
    cache_dir = '/tmp/model_cache'
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f'[Init] ✅ Cache directory ready: {cache_dir}')
    
    # Step 3: Pre-warm critical components (don't download full model yet)
    logger.info('[Init] 🔄 Pre-warming components...')
    
    # Test HuggingFace connection
    from huggingface_hub import whoami
    try:
        whoami()
        logger.info('[Init] ✅ HuggingFace Hub connection verified')
    except Exception as e:
        logger.info(f'[Init] 📡 HuggingFace Hub connection: {e} (will work during training)')
    
    # Step 4: Mark as ready
    Path('/tmp/ready').touch()
    if os.path.exists('/tmp/initializing'):
        os.remove('/tmp/initializing')
    
    logger.info('[Init] ✅ Worker ready to accept training jobs!')
    
except Exception as e:
    logger.error(f'[Init] ❌ Initialization failed: {e}')
    # Don't create ready flag - health will return 503/204
    sys.exit(1)
" &

INIT_PID=$!

# Wait for initialization to complete (with timeout)
echo "[Start] ⏳ Waiting for worker initialization..."
timeout=300  # 5 minutes max
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if [ -f "/tmp/ready" ]; then
        echo "[Start] ✅ Worker initialization completed!"
        break
    fi
    
    if ! kill -0 $INIT_PID 2>/dev/null; then
        echo "[Start] ❌ Initialization process failed"
        exit 1
    fi
    
    sleep 5
    elapsed=$((elapsed + 5))
    echo "[Start] 🔄 Initializing... ($elapsed/${timeout}s)"
done

if [ ! -f "/tmp/ready" ]; then
    echo "[Start] ❌ Initialization timeout after ${timeout}s"
    exit 1
fi

# Start the RunPod serverless handler (job processing loop)
echo "[Start] 🎯 Starting RunPod job handler..."
python /workspace/handler.py &
HANDLER_PID=$!

echo "[Start] ✅ All services started:"
echo "  - Health server (PID: $HEALTH_PID)"
echo "  - Job handler (PID: $HANDLER_PID)"

# Keep the container running and monitor processes
while true; do
    # Check if health server is still running
    if ! kill -0 $HEALTH_PID 2>/dev/null; then
        echo "[Start] ❌ Health server died, restarting..."
        python /workspace/healthcheck.py &
        HEALTH_PID=$!
    fi
    
    # Check if job handler is still running
    if ! kill -0 $HANDLER_PID 2>/dev/null; then
        echo "[Start] ❌ Job handler died, restarting..."
        python /workspace/handler.py &
        HANDLER_PID=$!
    fi
    
    sleep 30
done