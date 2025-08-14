#!/usr/bin/env python3
"""
Enhanced RunPod worker health check implementation
Based on official RunPod documentation recommendations
"""

from fastapi import FastAPI
import threading
import time
import logging
import os

# Global readiness state
ready = False
model_loaded = False
worker_connected = False

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/readyz")
def readyz():
    """
    Health check endpoint - only returns ready when all systems are operational
    """
    global ready, model_loaded, worker_connected
    
    status = {
        "status": "ready" if ready else "not ready",
        "model_loaded": model_loaded,
        "worker_connected": worker_connected,
        "timestamp": time.time()
    }
    
    if ready:
        return status
    else:
        # Return 503 Service Unavailable if not ready
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=status)

@app.get("/health")
def health():
    """
    Basic health endpoint - always responds if container is running
    """
    return {"status": "alive", "timestamp": time.time()}

def load_model():
    """
    Simulate model loading with proper error handling and retries
    """
    global model_loaded
    
    max_retries = 3
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model (attempt {attempt + 1}/{max_retries})...")
            
            # Simulate model loading time
            # In real implementation, this would load your Flux LoRA model
            time.sleep(30)  # Simulate heavy model loading
            
            logger.info("Model loaded successfully")
            model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Model loading failed after all retries")
                return False

def connect_to_queue():
    """
    Connect to RunPod queue with proper error handling
    """
    global worker_connected
    
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to queue (attempt {attempt + 1}/{max_retries})...")
            
            # In real implementation, connect to RunPod queue
            # This would include Redis connection or RunPod SDK initialization
            time.sleep(2)  # Simulate connection time
            
            logger.info("Queue connection established")
            worker_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Queue connection failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Queue connection failed after all retries")
                return False

def worker_loop():
    """
    Main worker loop - runs in background thread to avoid blocking health checks
    """
    logger.info("Worker loop started")
    
    while True:
        try:
            if not worker_connected:
                logger.warning("Worker not connected to queue, attempting reconnection...")
                if not connect_to_queue():
                    time.sleep(30)
                    continue
            
            if not model_loaded:
                logger.warning("Model not loaded, attempting reload...")
                if not load_model():
                    time.sleep(30)
                    continue
            
            # Process jobs from RunPod queue
            # In real implementation, this would use RunPod SDK:
            # job = runpod.get_job()
            # if job:
            #     result = process_job(job)
            #     runpod.post_result(result)
            
            logger.debug("Checking for jobs...")
            time.sleep(5)  # Poll every 5 seconds
            
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            time.sleep(10)  # Wait before retrying

def initialize_worker():
    """
    Initialize worker components in the correct order
    """
    global ready
    
    logger.info("Initializing worker...")
    
    # Step 1: Load model
    if not load_model():
        logger.error("Failed to load model during initialization")
        return False
    
    # Step 2: Connect to queue
    if not connect_to_queue():
        logger.error("Failed to connect to queue during initialization")
        return False
    
    # Step 3: Mark as ready
    ready = True
    logger.info("Worker initialization complete - marking as ready")
    return True

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event - initialize worker in background
    """
    logger.info("FastAPI startup - beginning worker initialization")
    
    # Run initialization in background thread to avoid blocking FastAPI
    def init_worker():
        initialize_worker()
    
    # Start worker loop in background thread
    def start_worker():
        worker_loop()
    
    # Start both threads
    threading.Thread(target=init_worker, daemon=True).start()
    threading.Thread(target=start_worker, daemon=True).start()
    
    logger.info("Background threads started")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.getenv("PORT", 8000))
    
    # Configure uvicorn with proper settings for RunPod
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )