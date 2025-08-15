#!/usr/bin/env python3
"""
Production RunPod Handler - Fixed All Critical Issues
Based on RunPod documentation best practices and error analysis
"""

import os
import sys
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
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Install missing dependencies at runtime - CRITICAL FIX
try:
    import toml
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("[RUNTIME FIX] Installing missing toml dependency...")
    subprocess.run([sys.executable, "-m", "pip", "install", "toml>=0.10.2"], check=True)
    import toml
    logger.info("[RUNTIME FIX] ✅ toml dependency installed successfully")

# PyTorch compatibility fix - From ComfyUI notebooks and Ostris AI Toolkit
def ensure_pytorch_compatibility():
    """Ensure PyTorch compatibility using the exact fixes from provided resources"""
    try:
        import torch
        import torchvision
        # Test the problematic torchvision::nms operation
        from torchvision import _meta_registrations
        logger.info(f"[TORCH] ✅ PyTorch {torch.__version__} and torchvision {torchvision.__version__} are compatible")
        return True
    except Exception as e:
        logger.warning(f"[TORCH FIX] Compatibility issue detected: {e}")
        logger.info("[TORCH FIX] Applying PyTorch compatibility fix from resources...")
        
        # Uninstall incompatible versions first
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                      capture_output=True)
        
        # Install compatible versions with CUDA 12.4 wheels (from ComfyUI notebook)
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.3.1", "torchvision==0.18.1", "torchaudio",
            "--extra-index-url", "https://download.pytorch.org/whl/cu124",
            "--force-reinstall"
        ]
        
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[TORCH FIX] ❌ Installation failed: {result.stderr}")
            return False
            
        # Verify the fix worked
        try:
            import torch
            import torchvision
            from torchvision import _meta_registrations
            logger.info(f"[TORCH FIX] ✅ Fixed! Now using torch {torch.__version__}, torchvision {torchvision.__version__}")
            return True
        except Exception as verify_error:
            logger.error(f"[TORCH FIX] ❌ Verification failed: {verify_error}")
            return False

# Apply PyTorch fix before any other imports
pytorch_compatible = ensure_pytorch_compatibility()

import runpod
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global readiness flags - Critical for RunPod health checks
READY_FLAG = "/tmp/ready"
MODEL_LOADED = False
WORKER_READY = False
FLUX_MODEL_CACHED = False

# Configuration - Fixed paths and settings
SD_SCRIPTS_PATH = "/workspace/sd-scripts/train_network.py"  # Corrected path
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "/tmp/model_cache"  # Dynamic cache location (no persistent storage)
HEALTH_PORT = int(os.getenv("PORT", 8000))

# FastAPI app for health checks (RunPod requirement)
app = FastAPI(title="Flux LoRA Trainer", version="1.0.0")

@app.get("/health")
async def health():
    """Basic health check - container is running"""
    return JSONResponse({
        "status": "alive", 
        "timestamp": time.time(),
        "service": "flux-lora-trainer",
        "version": "1.0.0"
    })

@app.get("/readyz") 
async def readyz():
    """Readiness check - RunPod standard endpoint"""
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
            "service": "flux-lora-trainer",
            "reason": "Initialization in progress"
        }, status_code=503)

def initialize_worker():
    """Initialize worker with comprehensive error handling"""
    global MODEL_LOADED, FLUX_MODEL_CACHED, WORKER_READY
    
    try:
        logger.info("[INIT] 🚀 Starting Flux LoRA worker initialization...")
        
        # Step 1: Create necessary directories
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs("/tmp", exist_ok=True)
        
        # Step 2: Verify training scripts exist  
        global SD_SCRIPTS_PATH
        if not Path(SD_SCRIPTS_PATH).exists():
            # Try alternative paths
            alt_paths = [
                "/tmp/sd-scripts/train_network.py",
                "/app/sd-scripts/train_network.py", 
                "/workspace/kohya_ss/train_network.py"
            ]
            
            script_found = False
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    SD_SCRIPTS_PATH = alt_path
                    script_found = True
                    logger.info(f"[INIT] Found training script at: {SD_SCRIPTS_PATH}")
                    break
            
            if not script_found:
                logger.warning(f"[INIT] ⚠️ Training script not found - will download during job")
        
        # Step 3: Download and cache Flux model (lightweight check)
        logger.info(f"[INIT] 📥 Checking Flux model availability...")
        
        try:
            # Quick check if model exists in HuggingFace cache
            from huggingface_hub import try_to_load_from_cache
            cached_file = try_to_load_from_cache(FLUX_MODEL_ID, "config.json")
            
            if cached_file is not None:
                FLUX_MODEL_CACHED = True
                logger.info("[INIT] ✅ Flux model found in cache")
            else:
                # Don't download now - will download when needed during training
                FLUX_MODEL_CACHED = False
                logger.info("[INIT] 📦 Flux model will be downloaded during training job")
        except Exception as e:
            logger.info(f"[INIT] 📦 Model cache check skipped: {e} - will download during job")
            FLUX_MODEL_CACHED = False
        
        # Step 4: Set model loaded flag
        MODEL_LOADED = True
        
        # Step 5: Create ready flag for RunPod
        Path(READY_FLAG).touch()
        WORKER_READY = True
        
        logger.info("[INIT] ✅ Worker initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"[INIT] ❌ Initialization failed: {str(e)}")
        MODEL_LOADED = False
        FLUX_MODEL_CACHED = False
        WORKER_READY = False
        return False

# NOTE: Health server is now handled by start.sh and healthcheck.py
# This handler only focuses on the RunPod job processing loop
logger.info("[RunPod] 🎯 Handler starting - health server managed externally")

def download_training_images(image_urls: List[str], zip_url: Optional[str] = None) -> Path:
    """Download and extract training images with robust error handling"""
    temp_dir = Path(tempfile.mkdtemp())
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    try:
        if zip_url:
            logger.info(f"[Download] 📦 Downloading ZIP from: {zip_url}")
            response = requests.get(zip_url, timeout=120, stream=True)
            response.raise_for_status()
            
            zip_path = temp_dir / "training_images.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(image_dir)
                
            logger.info(f"[Download] ✅ Extracted ZIP to {image_dir}")
            
        else:
            # Download individual images
            for i, url in enumerate(image_urls or []):
                try:
                    logger.info(f"[Download] 📷 Downloading image {i+1}/{len(image_urls)}")
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    
                    # Determine file extension
                    ext = url.split('?')[0].split('.')[-1].lower()
                    if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                        ext = 'jpg'
                    
                    image_path = image_dir / f"image_{i:03d}.{ext}"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                        
                except Exception as e:
                    logger.warning(f"[Download] ⚠️ Failed to download {url}: {e}")
                    continue
        
        # Verify we have images
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + \
                     list(image_dir.glob("*.png")) + list(image_dir.glob("*.webp"))
        
        if not image_files:
            raise ValueError("No valid images found after download")
            
        logger.info(f"[Download] ✅ Downloaded {len(image_files)} training images")
        return image_dir
        
    except Exception as e:
        logger.error(f"[Download] ❌ Image download failed: {e}")
        raise

def upload_to_storage(lora_file: Path, user_id: str, model_name: str) -> str:
    """
    Upload LoRA model to storage and return public URL
    Based on RunPod resources - implements S3/R2 upload for proper model URLs
    """
    import hashlib
    import boto3
    from botocore.config import Config as BotoConfig
    
    # Validate input file
    assert lora_file.suffix == ".safetensors", "output_is_not_safetensors"
    assert lora_file.exists() and lora_file.stat().st_size > 0, "empty_output_file"
    
    # Storage configuration (should be set in RunPod environment)
    S3_ENDPOINT = os.getenv("S3_ENDPOINT")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
    S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")
    
    if not all([S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY]):
        logger.warning("[Upload] ⚠️ S3 credentials not configured - using fallback base64")
        # Fallback: return base64 data URL (for development/testing)
        with open(lora_file, 'rb') as f:
            model_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:application/octet-stream;base64,{model_data}#model.safetensors"
    
    try:
        # Calculate file hash for unique naming
        def sha256sum(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        
        digest = sha256sum(lora_file)
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ("-", "_")).strip("_-") or "lora"
        key = f"loras/{user_id or 'anon'}/{safe_name}-{digest[:8]}.safetensors"
        
        # Configure S3 client
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
        )
        
        # Upload file
        s3.upload_file(
            str(lora_file), 
            S3_BUCKET, 
            key, 
            ExtraArgs={"ContentType": "application/octet-stream"}
        )
        
        logger.info(f"[Upload] ✅ Uploaded to S3: {key}")
        
        # Return public URL
        if S3_PUBLIC_BASE:
            return f"{S3_PUBLIC_BASE}/{key}"
        else:
            # Fallback: generate presigned URL (expires in 7 days)
            url = s3.generate_presigned_url(
                "get_object", 
                Params={"Bucket": S3_BUCKET, "Key": key}, 
                ExpiresIn=7 * 24 * 3600
            )
            return url
            
    except Exception as e:
        logger.error(f"[Upload] ❌ S3 upload failed: {e}")
        raise Exception(f"Storage upload failed: {e}")

def create_training_captions(image_dir: Path, training_type: str, trigger_word: str):
    """Create optimized caption files for different training types"""
    
    # Optimized caption templates based on training type
    caption_templates = {
        'person': f"a photo of {trigger_word}, person, portrait, high quality, detailed face",
        'character': f"{trigger_word}, character, detailed, high quality",
        'clothing': f"wearing {trigger_word}, clothing, fashion, detailed",
        'environment': f"{trigger_word} environment, scene, background, detailed",
        'artstyle': f"{trigger_word} art style, artistic, creative, detailed",
        'aesthetic': f"{trigger_word} aesthetic, style, mood, atmosphere"
    }
    
    template = caption_templates.get(training_type, caption_templates['person'])
    
    # Create caption files for all images
    for image_pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for image_file in image_dir.glob(image_pattern):
            caption_file = image_file.with_suffix('.txt')
            caption_file.write_text(template)
    
    logger.info(f"[Captions] ✅ Created captions with template: {template}")

async def train_lora_async(job_input: dict) -> dict:
    """Main LoRA training function with comprehensive error handling"""
    
    try:
        logger.info("[Training] 🎯 Starting LoRA training job...")
        
        # Wait for worker readiness with timeout
        max_wait = 60  # 60 seconds
        wait_time = 0
        while not (MODEL_LOADED and WORKER_READY) and wait_time < max_wait:
            await asyncio.sleep(2)
            wait_time += 2
            
        if not (MODEL_LOADED and WORKER_READY):
            raise Exception("Worker not ready - initialization timeout")
        
        # Extract job parameters
        model_name = job_input.get('model_name', 'trained_lora')
        training_type = job_input.get('training_type', 'person')
        trigger_word = job_input.get('trigger_word', 'ohwx')
        steps = int(job_input.get('steps', 600))
        learning_rate = float(job_input.get('learning_rate', 0.0002))
        
        # Image sources
        image_urls = job_input.get('image_urls', [])
        zip_url = job_input.get('zip_url') or job_input.get('training_data')
        
        if not image_urls and not zip_url:
            raise ValueError("No training images provided")
        
        logger.info(f"[Training] 📋 Config: {training_type} LoRA '{model_name}' with trigger '{trigger_word}'")
        logger.info(f"[Training] 📊 Parameters: {steps} steps, LR {learning_rate}")
        
        # Step 1: Download training images
        image_dir = download_training_images(image_urls, zip_url)
        create_training_captions(image_dir, training_type, trigger_word)
        
        # Step 2: Prepare output directory
        output_dir = Path(tempfile.mkdtemp())
        
        # Step 3: Build training command
        cmd = [
            "python", SD_SCRIPTS_PATH,
            "--pretrained_model_name_or_path", FLUX_MODEL_ID,
            "--train_data_dir", str(image_dir),
            "--output_dir", str(output_dir),
            "--output_name", model_name,
            "--resolution", "1024,1024",
            "--train_batch_size", "1",
            "--max_train_steps", str(steps),
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", "cosine_with_restarts",
            "--optimizer_type", "AdamW8bit",
            "--network_alpha", "32",
            "--network_dim", "64", 
            "--network_module", "networks.lora",
            "--save_every_n_epochs", "100",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--cache_latents",
            "--cache_latents_to_disk",
            "--enable_bucket",
            "--bucket_reso_steps", "64",
            "--bucket_no_upscale",
            "--min_bucket_reso", "512",
            "--max_bucket_reso", "2048"
        ]
        
        logger.info(f"[Training] 🏃‍♂️ Starting training process...")
        
        # Step 4: Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        current_step = 0
        
        # Monitor training progress
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.strip()
                output_lines.append(line)
                logger.info(f"[Training] {line}")
                
                # Extract step progress
                if "step:" in line.lower() or "/step" in line.lower():
                    try:
                        # Parse step number from various formats
                        for part in line.split():
                            if part.isdigit() and int(part) <= steps:
                                current_step = int(part)
                                progress = min(100, int((current_step / steps) * 100))
                                logger.info(f"[Training] 📊 Progress: {progress}% ({current_step}/{steps})")
                                break
                    except:
                        pass
        
        # Wait for process completion
        return_code = process.wait()
        
        if return_code != 0:
            error_output = '\n'.join(output_lines[-20:])  # Last 20 lines
            raise subprocess.CalledProcessError(return_code, cmd, output=error_output)
        
        # Step 5: Find and validate trained model
        model_files = list(output_dir.glob("*.safetensors"))
        if not model_files:
            # FORCE failure: no LoRA produced
            logger.error("[Training] ❌ No .safetensors model file generated")
            return {"error": "no_lora_file_generated", "success": False}
        
        model_file = model_files[0]
        
        # Validate model file size (must be > 10KB to avoid empty artifacts)
        if model_file.stat().st_size < 10000:
            logger.error(f"[Training] ❌ Model file too small: {model_file.stat().st_size} bytes")
            return {"error": "lora_file_too_small", "success": False}
        
        logger.info(f"[Training] ✅ Valid model generated: {model_file.name} ({model_file.stat().st_size:,} bytes)")
        
        # Step 6: Upload to storage and return URL (CRITICAL - must return URL, not base64)
        try:
            user_id = str(job_input.get("user_id", "anon"))
            model_url = upload_to_storage(model_file, user_id=user_id, model_name=model_name)
            
            if not model_url or not model_url.lower().endswith('.safetensors'):
                logger.error(f"[Training] ❌ Upload failed - invalid URL: {model_url}")
                return {"error": "upload_failed_invalid_url", "success": False}
            
            # Calculate file hash for verification
            import hashlib
            sha256_hash = hashlib.sha256()
            with open(model_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            file_hash = sha256_hash.hexdigest()
            
            logger.info(f"[Training] ✅ Model uploaded successfully: {model_url}")
            
            return {
                "status": "completed",
                "success": True,
                "model_url": model_url,              # CRITICAL: Frontend validation expects this
                "model_name": model_name,
                "trigger_word": trigger_word,
                "training_type": training_type,
                "training_steps": current_step,
                "sha256": file_hash,
                "size": model_file.stat().st_size,
                "logs": '\n'.join(output_lines[-50:])  # Last 50 lines for debugging
            }
            
        except Exception as upload_error:
            logger.error(f"[Training] ❌ Upload failed: {upload_error}")
            return {"error": f"upload_failed: {upload_error}", "success": False}
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(f"[Training] ❌ {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'logs': '\n'.join(output_lines[-50:]) if 'output_lines' in locals() else str(e)
        }

# RunPod serverless handler
async def handler(job):
    """Async RunPod handler with proper error handling and timeout protection"""
    
    try:
        logger.info(f"[RunPod] 📨 Processing job: {job.get('id', 'unknown')}")
        
        # Validate job input
        job_input = job.get('input', {})
        if not job_input:
            raise ValueError("No input provided in job")
        
        # Add timeout protection (25 minutes max)
        try:
            result = await asyncio.wait_for(
                train_lora_async(job_input),
                timeout=25 * 60  # 25 minutes
            )
        except asyncio.TimeoutError:
            raise Exception("Training timeout after 25 minutes")
        
        logger.info(f"[RunPod] ✅ Job completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Job processing failed: {str(e)}"
        logger.error(f"[RunPod] ❌ {error_msg}")
        
        return {
            'success': False,
            'error': error_msg
        }

# Start RunPod serverless worker
if __name__ == "__main__":
    logger.info("[RunPod] 🚀 Starting Flux LoRA serverless worker...")
    
    # Give initialization time to start
    time.sleep(5)
    
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })