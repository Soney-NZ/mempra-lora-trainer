#!/usr/bin/env python3
"""
Production Flux LoRA Training Handler for RunPod Serverless
Adapted from SDXL best practices for optimal Flux LoRA training
"""

import os
import json
import requests
import zipfile
import tempfile
import subprocess
import base64
from pathlib import Path
import threading
import asyncio
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

import runpod
from huggingface_hub import snapshot_download

# Global readiness flags
READY_FLAG = "/tmp/ready"
MODEL_LOADED = False
WORKER_READY = False
FLUX_MODEL_CACHED = False

# Configuration based on RunPod best practices
SD_SCRIPTS_PATH = "/tmp/sd-scripts/train_network.py"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "/tmp/model_cache"
HEALTH_CHECK_PORT = int(os.getenv("PORT", 8000))

# FastAPI app for health checks (RunPod requirement)
app = FastAPI()

@app.get("/health")
def health():
    """Basic health check - container is running"""
    return JSONResponse({
        "status": "alive", 
        "timestamp": time.time(),
        "service": "flux-lora-trainer"
    })

@app.get("/readyz")
def readyz():
    """Readiness check - worker is ready for jobs (RunPod standard)"""
    global MODEL_LOADED, WORKER_READY, FLUX_MODEL_CACHED
    
    if WORKER_READY and MODEL_LOADED and FLUX_MODEL_CACHED:
        return JSONResponse({
            "status": "ready",
            "model_loaded": MODEL_LOADED,
            "worker_ready": WORKER_READY,
            "flux_cached": FLUX_MODEL_CACHED,
            "timestamp": time.time(),
            "service": "flux-lora-trainer"
        })
    else:
        return JSONResponse({
            "status": "not ready",
            "model_loaded": MODEL_LOADED,
            "worker_ready": WORKER_READY,
            "flux_cached": FLUX_MODEL_CACHED,
            "timestamp": time.time(),
            "service": "flux-lora-trainer"
        }, status_code=503)

def initialize_flux_model():
    """Initialize Flux model with proper caching (following SDXL pattern)"""
    global MODEL_LOADED, FLUX_MODEL_CACHED, WORKER_READY
    
    try:
        print("[INIT] Starting Flux model initialization...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Download Flux model to cache
        print(f"[INIT] Downloading Flux model to {CACHE_DIR}...")
        snapshot_download(
            repo_id=FLUX_MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        
        FLUX_MODEL_CACHED = True
        MODEL_LOADED = True
        print("[INIT] ✅ Flux model cached successfully")
        
        # Create ready flag for RunPod health checks
        Path(READY_FLAG).touch()
        WORKER_READY = True
        print("[INIT] ✅ Worker ready flag set")
        
        return True
        
    except Exception as e:
        print(f"[INIT] ❌ Flux initialization failed: {str(e)}")
        FLUX_MODEL_CACHED = False
        MODEL_LOADED = False
        WORKER_READY = False
        return False

def start_health_server():
    """Start health check server (RunPod requirement)"""
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=HEALTH_CHECK_PORT, 
            log_level="info"
        )
    except Exception as e:
        print(f"[Flux LoRA] Health server error: {e}")

# Start initialization and health server in background threads
print("[Flux LoRA] Starting background services...")
threading.Thread(target=initialize_flux_model, daemon=True).start()
threading.Thread(target=start_health_server, daemon=True).start()

def download_training_images(image_urls: List[str], zip_url: Optional[str] = None) -> Path:
    """Download and prepare training images (adapted from SDXL pattern)"""
    temp_dir = Path(tempfile.mkdtemp())
    image_dir = temp_dir / "training_images"
    image_dir.mkdir(exist_ok=True)

    if zip_url:
        print(f"[Flux LoRA] Downloading image zip: {zip_url}")
        response = requests.get(zip_url, timeout=120)
        response.raise_for_status()
        zip_path = temp_dir / "images.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        
        print(f"[Flux LoRA] Extracted {len(list(image_dir.glob('*')))} files from zip")
    
    else:
        print(f"[Flux LoRA] Downloading {len(image_urls)} individual images...")
        for i, url in enumerate(image_urls or []):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # Determine file extension
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                elif 'png' in content_type:
                    ext = 'png'
                elif 'webp' in content_type:
                    ext = 'webp'
                else:
                    # Fallback: try to get from URL
                    ext = url.split('?')[0].split('.')[-1].lower()
                    if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                        ext = 'jpg'
                
                image_path = image_dir / f"image_{i:03d}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                    
                print(f"[Flux LoRA] Downloaded image {i+1}/{len(image_urls)}")
                
            except Exception as e:
                print(f"[Flux LoRA] Failed to download {url}: {e}")
    
    downloaded_count = len(list(image_dir.glob('*.*')))
    print(f"[Flux LoRA] Total images prepared: {downloaded_count}")
    return image_dir

def create_training_captions(image_dir: Path, training_type: str, trigger_word: str):
    """Create caption files optimized for Flux LoRA training"""
    # Flux-optimized caption templates
    caption_templates = {
        'person': f"a photo of {trigger_word}, person, portrait, detailed, high quality",
        'character': f"{trigger_word}, character, detailed, high quality",
        'style': f"in the style of {trigger_word}, artistic style, detailed",
        'object': f"a {trigger_word}, object, detailed, high quality",
        'clothing': f"wearing {trigger_word}, clothing, detailed, high quality"
    }
    
    template = caption_templates.get(training_type, caption_templates['person'])
    caption_count = 0
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for image_file in image_dir.glob(ext):
            caption_file = image_file.with_suffix('.txt')
            caption_file.write_text(template)
            caption_count += 1
    
    print(f"[Flux LoRA] Created {caption_count} caption files")

def execute_flux_lora_training(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Flux LoRA training with optimized parameters"""
    
    # Extract parameters with defaults optimized for Flux
    model_name = job_input.get('model_name', 'flux_lora_model')
    training_type = job_input.get('training_type', 'person')
    trigger_word = job_input.get('trigger_word', 'ohwx')
    steps = job_input.get('steps', 600)  # Optimal for Flux LoRA
    learning_rate = job_input.get('learning_rate', 0.0002)  # BFL recommended
    batch_size = job_input.get('batch_size', 1)
    resolution = job_input.get('resolution', '512,512')
    
    # Image sources
    image_urls = job_input.get('image_urls', [])
    zip_url = job_input.get('zip_url')
    
    if not image_urls and not zip_url:
        raise ValueError("No training images provided")

    print(f"[Flux LoRA] Starting training: {model_name} ({training_type})")
    print(f"[Flux LoRA] Trigger word: '{trigger_word}'")
    print(f"[Flux LoRA] Parameters: {steps} steps, LR {learning_rate}")
    
    # Download and prepare images
    image_dir = download_training_images(image_urls, zip_url)
    create_training_captions(image_dir, training_type, trigger_word)
    
    # Setup output directory
    output_dir = Path(tempfile.mkdtemp(prefix="flux_lora_"))
    
    # Build training command with Flux-optimized parameters
    training_cmd = [
        "python", SD_SCRIPTS_PATH,
        "--pretrained_model_name_or_path", FLUX_MODEL_ID,
        "--train_data_dir", str(image_dir),
        "--output_dir", str(output_dir),
        "--resolution", resolution,
        "--train_batch_size", str(batch_size),
        "--max_train_steps", str(steps),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "cosine_with_restarts",
        "--optimizer_type", "AdamW8bit",
        "--network_alpha", "1.0",
        "--network_dim", "64",  # Optimal for Flux
        "--network_module", "networks.lora",
        "--save_every_n_epochs", "1",
        "--mixed_precision", "bf16",
        "--save_precision", "bf16",
        "--cache_latents",
        "--enable_bucket",
        "--bucket_reso_steps", "64",
        "--bucket_no_upscale",
        "--gradient_checkpointing",
        "--xformers"  # For memory efficiency
    ]
    
    # Add type-specific optimizations
    if training_type == 'person':
        training_cmd.extend([
            "--face_crop_aug_range", "0.1,0.3",
            "--color_aug"
        ])
    elif training_type == 'style':
        training_cmd.extend([
            "--color_aug",
            "--flip_aug",
            "--random_crop"
        ])
    
    print(f"[Flux LoRA] Executing training command...")
    
    # Execute training with real-time monitoring
    training_logs = []
    process = subprocess.Popen(
        training_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    step_count = 0
    for line in process.stdout:
        line = line.strip()
        if line:
            training_logs.append(line)
            print(f"[Training] {line}")
            
            # Monitor progress
            if "step:" in line.lower() or "epoch" in line.lower():
                step_count += 1
                if step_count % 50 == 0:  # Progress every 50 steps
                    progress = min(100, int((step_count / steps) * 100))
                    print(f"[Flux LoRA] Progress: {progress}% (step {step_count})")
    
    process.wait()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, 
            " ".join(training_cmd[:3]) + "..."
        )
    
    # Find generated model files
    model_files = list(output_dir.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError("No .safetensors model file generated")
    
    # Get the trained model file
    model_file = model_files[-1]  # Latest model
    model_size = model_file.stat().st_size
    
    print(f"[Flux LoRA] Training completed successfully")
    print(f"[Flux LoRA] Model: {model_file.name} ({model_size:,} bytes)")
    
    # Read model data for output (following SDXL pattern)
    with open(model_file, 'rb') as f:
        model_data = f.read()
    
    # Encode as base64 (RunPod standard format)
    model_b64 = base64.b64encode(model_data).decode('utf-8')
    
    return {
        "model_name": model_name,
        "trigger_word": trigger_word,
        "training_type": training_type,
        "model_data_b64": model_b64,
        "model_size": model_size,
        "steps_completed": steps,
        "learning_rate": learning_rate,
        "training_logs": training_logs[-100:],  # Last 100 lines
        "status": "COMPLETED"
    }

async def handler(job):
    """
    Main RunPod handler function for Flux LoRA training
    Follows RunPod Serverless patterns from SDXL tutorial
    """
    job_id = job.get('id', 'unknown')
    print(f"[Flux LoRA] Processing job: {job_id}")
    
    try:
        # Verify worker readiness (critical for model training)
        if not (MODEL_LOADED and WORKER_READY and FLUX_MODEL_CACHED):
            print("[Flux LoRA] Worker not ready - waiting for initialization...")
            
            # Wait for initialization (up to 2 minutes)
            for i in range(120):
                if MODEL_LOADED and WORKER_READY and FLUX_MODEL_CACHED:
                    break
                await asyncio.sleep(1)
                if i % 30 == 0:  # Log every 30 seconds
                    print(f"[Flux LoRA] Still waiting for initialization... ({i}s)")
            
            if not (MODEL_LOADED and WORKER_READY and FLUX_MODEL_CACHED):
                raise Exception("Worker initialization timeout - model not ready")
        
        print(f"[Flux LoRA] Worker ready - starting LoRA training")
        
        # Extract job input
        job_input = job.get('input', {})
        
        # Execute training in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            execute_flux_lora_training, 
            job_input
        )
        
        print(f"[Flux LoRA] Job {job_id} completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Flux LoRA training failed: {str(e)}"
        print(f"[Flux LoRA] ERROR: {error_msg}")
        
        # Return proper error response (critical for worker health)
        return {
            "error": error_msg,
            "status": "FAILED",
            "job_id": job_id,
            "timestamp": time.time()
        }

def start_health_server():
    """Start health check server (RunPod requirement)"""
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=HEALTH_CHECK_PORT, access_log=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"[Health] Server started on port {HEALTH_CHECK_PORT}")

def main():
    """Main initialization - critical for worker health"""
    print("[Main] Starting Flux LoRA Worker...")
    
    # Start health server immediately (prevents unhealthy status)
    start_health_server()
    
    # Initialize Flux model in background
    init_thread = threading.Thread(target=initialize_flux_model, daemon=True)
    init_thread.start()
    
    print("[Main] Health server running, model initializing...")
    print("[Main] Starting RunPod serverless worker...")
    
    # Start RunPod worker with proper error handling
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
        "rp_log_level": "INFO"
    })

if __name__ == "__main__":
    main()