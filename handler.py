    import os
    import json
    import requests
    import zipfile
    import tempfile
    import subprocess
    from pathlib import Path
    import threading
    from typing import List, Optional

    import runpod
    from huggingface_hub import snapshot_download

    READY_FLAG = "/tmp/ready"
    SD_SCRIPTS_PATH = "/tmp/sd-scripts/train_network.py"
    HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"

    def _touch_ready():
        Path(READY_FLAG).parent.mkdir(parents=True, exist_ok=True)
        Path(READY_FLAG).write_text("ok")

    def _warmup():
        try:
            assert Path(SD_SCRIPTS_PATH).exists(), f"Missing {SD_SCRIPTS_PATH}"
            snapshot_download(repo_id=HF_MODEL_ID, local_dir=None, local_dir_use_symlinks=True)
        except Exception as e:
            print(f"[warmup] non-fatal: {e}")
        finally:
            _touch_ready()

    threading.Thread(target=_warmup, daemon=True).start()

    def download_and_extract_images(image_urls: List[str], zip_url: Optional[str] = None) -> Path:
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
        try:
            model_name = job_input.get('model_name', 'trained_lora')
            trigger_word = job_input.get('trigger_word', 'special')
            training_steps = int(job_input.get('training_steps', 1200))
            learning_rate = float(job_input.get('learning_rate', 0.0001))
            network_dim = int(job_input.get('network_dim', 32))
            training_type = job_input.get('training_type', 'person')

            image_urls = job_input.get('images', []) or []
            zip_url = job_input.get('zip_url')
            if not image_urls and not zip_url:
                return {"error": "No training images provided"}

            print(f"Starting LoRA training for {model_name}")
            image_dir = download_and_extract_images(image_urls, zip_url)
            create_captions(image_dir, training_type, trigger_word)

            output_dir = Path("/tmp/lora_output")
            output_dir.mkdir(exist_ok=True)

            cmd = [
                "python", SD_SCRIPTS_PATH,
                f"--pretrained_model_name_or_path={HF_MODEL_ID}",
                f"--train_data_dir={image_dir}",
                f"--output_dir={output_dir}",
                f"--output_name={model_name}",
                "--network_module=networks.lora",
                f"--network_dim={network_dim}",
                f"--network_alpha={max(1, network_dim // 2)}",
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

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)

            if result.returncode != 0:
                return {"error": f"Training failed: {result.stderr}", "stdout": result.stdout[-2000:]}

            lora_files = sorted(output_dir.glob("*.safetensors"))
            if not lora_files:
                return {"error": "No LoRA file generated"}

            return {
                "status": "completed",
                "model_path": str(lora_files[-1]),
                "model_name": model_name,
                "trigger_word": trigger_word,
                "training_steps": training_steps,
                "logs": result.stdout[-2000:]
            }
        except Exception as e:
            return {"error": f"Training error: {str(e)}"}

    def handler(job):
        job_input = job.get("input", {})
        try:
            return train_lora(job_input)
        except Exception as e:
            return {"error": f"Handler error: {str(e)}"}

    if __name__ == "__main__":
        if not Path(READY_FLAG).exists():
            _touch_ready()
        runpod.serverless.start({"handler": handler})