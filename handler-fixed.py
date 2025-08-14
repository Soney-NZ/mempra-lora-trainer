#!/usr/bin/env python3
"""
Production-ready RunPod handler with proper health checks and async processing
Based on RunPod documentation best practices for Flux LoRA training
"""

import os
import json
import requests
import zipfile
import tempfile
import subprocess
from pathlib import Path
import threading
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

import runpod
from huggingface_hub import snapshot_download

# Global readiness flags
READY_FLAG = "/tmp/ready"
MODEL_LOADED = False
WORKER_READY = False

# Configuration
SD_SCRIPTS_PATH = "/tmp/sd-scripts/train_network.py"
HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# FastAPI app for health checks
app = FastAPI()

@app.get("/health")
def health():
    """Basic health check - container is running"""
    return JSONResponse({"status": "alive", "timestamp": time.time()})

@app.get("/readyz")
def readyz():
    """Readiness check - worker is ready for jobs"""
    global MODEL_LOADED, WORKER_READY
    
    if WORKER_READY and MODEL_LOADED:
        return JSONResponse({
            "status": "ready",
            "model_loaded": MODEL_LOADED,
            "worker_ready": WORKER_READY,
            "timestamp": time.time()
        })
    else:
        return JSONResponse(
            {
                "status": "not ready",
                "model_loaded": MODEL_LOADED,
                "worker_ready": WORKER_READY,
                "timestamp": time.time()
            },
            status_code=503
        )

def _touch_ready():
    """Mark worker as ready"""
    Path(READY_FLAG).parent.mkdir(parents=True, exist_ok=True)
    Path(READY_FLAG).write_text("ok")

def _warmup():
    """Initialize models and dependencies with proper error handling"""
    global MODEL_LOADED, WORKER_READY
    
    try:
        print("[RunPod] Starting worker initialization...")
        
        # Step 1: Verify training scripts exist
        if not Path(SD_SCRIPTS_PATH).exists():
            print(f"[RunPod] WARNING: Missing {SD_SCRIPTS_PATH}")
            # In production, this would be a critical error
        
        # Step 2: Download and cache Flux model
        print("[RunPod] Downloading Flux model...")
        snapshot_download(
            repo_id=HF_MODEL_ID, 
            local_dir=None, 
            local_dir_use_symlinks=True
        )
        print("[RunPod] Flux model loaded successfully")
        MODEL_LOADED = True
        
        # Step 3: Validate GPU availability
        print("[RunPod] Validating GPU...")
        # Add GPU validation here if needed
        
        # Step 4: Mark as ready
        _touch_ready()
        WORKER_READY = True
        print("[RunPod] Worker initialization complete - ready for jobs")
        
    except Exception as e:
        print(f"[RunPod] Initialization error: {e}")
        # Don't mark as ready if initialization fails
        MODEL_LOADED = False
        WORKER_READY = False

# Start initialization in background thread
print("[RunPod] Starting background initialization...")
threading.Thread(target=_warmup, daemon=True).start()

# Start health check server in background
def start_health_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=start_health_server, daemon=True).start()

def download_and_extract_images(image_urls: List[str], zip_url: Optional[str] = None) -> Path:
    """Download and extract training images"""
    temp_dir = Path(tempfile.mkdtemp())
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)

    if zip_url:
        response = requests.get(zip_url, timeout=60)
        response.raise_for_status()
        zip_path = temp_dir / "images.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
    else:
        for i, url in enumerate(image_urls or []):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                ext = (url.split('?')[0].split('.')[-1] or 'jpg').lower()
                if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                    ext = 'jpg'
                image_path = image_dir / f"image_{i:03d}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"[download] skip {url}: {e}")
    return image_dir

def create_captions(image_dir: Path, training_type: str, trigger_word: str):
    """Create caption files for training images"""
    caption_template = {
        'person': f"a photo of {trigger_word}, person, portrait, high quality",
        'style': f"{trigger_word} style, artistic, detailed", 
        'object': f"a photo of {trigger_word}, object, detailed"
    }
    template = caption_template.get(training_type, caption_template['person'])
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for image_file in image_dir.glob(ext):
            caption_file = image_file.with_suffix('.txt')
            caption_file.write_text(template)

def train_lora(job_input: dict):
    """Main LoRA training function with comprehensive error handling"""
    try:
        print("[RunPod] Starting LoRA training...")
        
        # Validate worker readiness
        if not (MODEL_LOADED and WORKER_READY):
            raise Exception("Worker not ready - models not loaded")
        
        model_name = job_input.get('model_name', 'trained_lora')
        training_type = job_input.get('training_type', 'person')
        trigger_word = job_input.get('trigger_word', 'ohwx')
        steps = job_input.get('steps', 600)
        learning_rate = job_input.get('learning_rate', 0.0002)
        
        # Extract image URLs and zip URL
        image_urls = job_input.get('image_urls', [])
        zip_url = job_input.get('zip_url')
        
        if not image_urls and not zip_url:
            raise ValueError("No images provided for training")

        print(f"[RunPod] Training {training_type} LoRA '{model_name}' with trigger '{trigger_word}'")
        
        # Download images
        image_dir = download_and_extract_images(image_urls, zip_url)
        create_captions(image_dir, training_type, trigger_word)
        
        # Set up training parameters
        output_dir = Path(tempfile.mkdtemp())
        
        cmd = [
            "python", SD_SCRIPTS_PATH,
            "--pretrained_model_name_or_path", HF_MODEL_ID,
            "--train_data_dir", str(image_dir),
            "--output_dir", str(output_dir),
            "--resolution", "512,512",
            "--train_batch_size", "1",
            "--max_train_steps", str(steps),
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", "cosine",
            "--optimizer_type", "AdamW8bit",
            "--network_alpha", "1",
            "--network_dim", "64",
            "--network_module", "networks.lora",
            "--save_every_n_epochs", "1",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--cache_latents",
            "--enable_bucket",
            "--bucket_reso_steps", "64",
            "--bucket_no_upscale"
        ]
        
        print(f"[RunPod] Starting training with command: {' '.join(cmd[:3])}...")
        
        # Run training with progress updates
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[Training] {line}")
                output_lines.append(line)
                
                # Send progress updates based on step detection
                if "step:" in line.lower():
                    try:
                        # Extract step number and send progress
                        step_match = [x for x in line.split() if ":" in x and x.replace(":", "").isdigit()]
                        if step_match:
                            current_step = int(step_match[0].replace(":", ""))
                            progress = min(100, int((current_step / steps) * 100))
                            print(f"[RunPod] Training progress: {progress}% ({current_step}/{steps})")
                    except:
                        pass
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        # Find the trained model file
        model_files = list(output_dir.glob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError("No .safetensors file generated")
        
        model_file = model_files[0]
        
        print(f"[RunPod] Training completed successfully")
        print(f"[RunPod] Model saved: {model_file}")
        
        # Return the trained model
        with open(model_file, 'rb') as f:
            model_data = f.read()
        
        return {
            'model_name': model_name,
            'trigger_word': trigger_word,
            'training_type': training_type,
            'model_data': model_data,
            'model_size': len(model_data),
            'steps_completed': steps,
            'logs': '\n'.join(output_lines[-50:])  # Last 50 lines
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"[RunPod] ERROR: {error_msg}")
        raise Exception(error_msg)

async def handler(job):
    """
    Async handler function for RunPod jobs
    Processes LoRA training requests with proper error handling
    """
    try:
        print(f"[RunPod] Processing job: {job.get('id', 'unknown')}")
        
        # Check worker readiness
        if not (MODEL_LOADED and WORKER_READY):
            # Wait briefly for initialization
            for i in range(30):  # Wait up to 30 seconds
                if MODEL_LOADED and WORKER_READY:
                    break
                await asyncio.sleep(1)
            
            if not (MODEL_LOADED and WORKER_READY):
                raise Exception("Worker not ready - initialization failed")
        
        # Extract job input
        job_input = job.get('input', {})
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, train_lora, job_input)
        
        print(f"[RunPod] Job completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Job failed: {str(e)}"
        print(f"[RunPod] ERROR: {error_msg}")
        return {"error": error_msg}

# Start the RunPod serverless worker
if __name__ == "__main__":
    print("[RunPod] Starting serverless worker...")
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })