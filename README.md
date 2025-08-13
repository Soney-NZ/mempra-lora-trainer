# RunPod LoRA Training Setup

This creates a custom RunPod endpoint for training LoRA models using Kohya scripts.

## Setup Steps:

### 1. Create GitHub Repository

1. Create a new GitHub repository (e.g., `lora-trainer`)
2. Upload these files to the repository:
   - `handler.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`

### 2. Create RunPod Endpoint

1. Go to https://console.runpod.io/serverless/new-endpoint
2. Choose "Import Git Repository"
3. Enter your GitHub repository URL: `https://github.com/yourusername/lora-trainer`
4. Set endpoint name: `flux-lora-trainer`
5. Configure:
   - GPU: RTX A6000 or A100
   - Container Disk: 20GB
   - Max Workers: 1-3
   - Python Version: 3.10
6. Deploy endpoint

### 3. Get Endpoint ID

After deployment, copy the endpoint ID (format: `abc123def-456-789`)

### 4. Update Replit Environment

Add to your Replit Secrets:
- `RUNPOD_LORA_ENDPOINT_ID` = your endpoint ID

## Input Format:

```json
{
  "model_name": "My AI Twin",
  "trigger_word": "MyFACE", 
  "training_steps": 1200,
  "learning_rate": 0.0001,
  "network_dim": 32,
  "training_type": "person",
  "images": ["url1", "url2", ...],
  "zip_url": "optional_zip_url"
}
```

## Output Format:

```json
{
  "status": "completed",
  "model_path": "/path/to/model.safetensors",
  "model_name": "My AI Twin",
  "trigger_word": "MyFACE",
  "training_steps": 1200,
  "logs": "training logs..."
}
```

This will provide real GPU training with Kohya for high-quality LoRA models.