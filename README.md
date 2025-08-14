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
5. **CRITICAL - Configure GPU Settings:**
   - **GPU Type**: RTX 4090 (24GB) or RTX A6000 (48GB) for LoRA training
   - **Container Disk**: 20GB minimum
   - **Max Workers**: 1-3 (based on demand)
   - **Worker Idle Timeout**: 5 seconds
   - **Flash Boot**: Enabled (faster startup)
   - **GPU Count**: 1 (LoRA doesn't need multi-GPU)
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

## GPU Queue Management:

If training gets stuck "IN_QUEUE":
1. **Check GPU Availability**: RunPod shows real-time GPU availability
2. **Switch GPU Type**: Try RTX 4090 instead of A6000 (often more available)
3. **Monitor Queue**: High demand periods may require waiting
4. **Use Community Cloud**: Often faster than Secure Cloud for LoRA training

## Troubleshooting:

- **"IN_QUEUE" for >5 minutes**: GPU demand is high, try different GPU type
- **"FAILED" status**: Check logs for CUDA memory errors or dependency issues
- **No model output**: Verify ZIP upload and trigger word format

This provides real GPU training with Kohya for high-quality LoRA models optimized for Flux generation.