import os
import json
import requests
import zipfile
import tempfile
import subprocess
from pathlib import Path
import runpod

def download_and_extract_images(image_urls, zip_url=None):
    """Download and extract training images"""
    temp_dir = Path(tempfile.mkdtemp())
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    if zip_url:
        # Download ZIP file
        response = requests.get(zip_url)
        zip_path = temp_dir / "images.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
    else:
        # Download individual images
        for i, url in enumerate(image_urls):
            response = requests.get(url)
            if response.status_code == 200:
                ext = url.split('.')[-1].lower()
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
                image_path = image_dir / f"image_{i:03d}.{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
    
    return image_dir

def create_captions(image_dir, training_type, trigger_word):
    """Create caption files for training images"""
    caption_template = {
        'person': f"a photo of {trigger_word}, person, portrait, high quality",
        'style': f"{trigger_word} style, artistic, detailed",
        'object': f"a photo of {trigger_word}, object, detailed"
    }
    
    template = caption_template.get(training_type, caption_template['person'])
    
    for image_file in image_dir.glob("*.{jpg,jpeg,png}"):
        caption_file = image_file.with_suffix('.txt')
        with open(caption_file, 'w') as f:
            f.write(template)

def train_lora(job_input):
    """Main training function"""
    try:
        # Parse input
        model_name = job_input.get('model_name', 'trained_lora')
        trigger_word = job_input.get('trigger_word', 'special')
        training_steps = job_input.get('training_steps', 1200)
        learning_rate = job_input.get('learning_rate', 0.0001)
        network_dim = job_input.get('network_dim', 32)
        training_type = job_input.get('training_type', 'person')
        
        # Image data
        image_urls = job_input.get('images', [])
        zip_url = job_input.get('zip_url')
        
        if not image_urls and not zip_url:
            return {"error": "No training images provided"}
        
        print(f"Starting LoRA training for {model_name}")
        print(f"Training type: {training_type}")
        print(f"Images: {len(image_urls) if image_urls else 'ZIP file'}")
        
        # Download and prepare images
        image_dir = download_and_extract_images(image_urls, zip_url)
        create_captions(image_dir, training_type, trigger_word)
        
        # Setup output directory
        output_dir = Path("/tmp/lora_output")
        output_dir.mkdir(exist_ok=True)
        
        # Use pre-installed Kohya training scripts
        print("Using pre-installed Kohya training scripts...")
        
        # Kohya training command
        cmd = [
            "python", "/tmp/sd-scripts/train_network.py",
            "--pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev",
            f"--train_data_dir={image_dir}",
            f"--output_dir={output_dir}",
            f"--output_name={model_name}",
            "--network_module=networks.lora",
            f"--network_dim={network_dim}",
            f"--network_alpha={network_dim // 2}",
            f"--learning_rate={learning_rate}",
            f"--max_train_steps={training_steps}",
            "--resolution=1024,1024",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            "--mixed_precision=fp16",
            "--save_precision=fp16",
            "--optimizer_type=AdamW8bit",
            "--lr_scheduler=cosine",
            "--lr_warmup_steps=100",
            "--save_every_n_steps=500",
            "--sample_every_n_steps=250",
            "--logging_dir=/tmp/logs",
            "--cache_latents",
            "--cache_latents_to_disk"
        ]
        
        # Type-specific optimizations
        if training_type == 'person':
            cmd.extend([
                "--enable_bucket",
                "--bucket_reso_steps=64",
                "--face_crop_aug_range=0.1,0.4"
            ])
        elif training_type == 'style':
            cmd.extend([
                "--color_aug",
                "--flip_aug",
                "--random_crop"
            ])
        
        print("Starting Kohya training...")
        print(f"Command: {' '.join(cmd)}")
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return {
                "error": f"Training failed: {result.stderr}",
                "stdout": result.stdout
            }
        
        # Find output file
        lora_files = list(output_dir.glob("*.safetensors"))
        if not lora_files:
            return {"error": "No LoRA file generated"}
        
        lora_file = lora_files[0]
        
        return {
            "status": "completed",
            "model_path": str(lora_file),
            "model_name": model_name,
            "trigger_word": trigger_word,
            "training_steps": training_steps,
            "logs": result.stdout[-1000:]  # Last 1000 chars of logs
        }

        
    except Exception as e:
        return {"error": f"Training error: {str(e)}"}

def handler(job):
    """RunPod handler function"""
    job_input = job.get("input", {})
    
    try:
        result = train_lora(job_input)
        return result
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})