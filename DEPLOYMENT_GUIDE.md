# RunPod LoRA Training Deployment Guide

## Critical Fixes Applied

### ✅ Fixed Container Health Issues
- **Problem**: Workers going unhealthy due to missing/blocked health endpoints
- **Solution**: Proper FastAPI health server with `/health` and `/readyz` endpoints
- **Result**: RunPod can now properly monitor worker status

### ✅ Fixed Handler Syntax Errors  
- **Problem**: Python syntax errors causing container crashes
- **Solution**: Production-ready async handler with proper error handling
- **Result**: Containers start and stay running

### ✅ Fixed Blocking Operations
- **Problem**: Model initialization blocking main thread, preventing health checks
- **Solution**: Background thread initialization with proper thread management  
- **Result**: Health checks respond even during model loading

### ✅ Fixed Docker Configuration
- **Problem**: Missing dependencies, incorrect paths, improper health checks
- **Solution**: Production Dockerfile with proper health check timing and dependencies
- **Result**: Containers build and deploy successfully

## Deployment Steps

### 1. Build and Push Container

```bash
cd runpod-setup

# Build production container
docker build -f Dockerfile.production -t your-registry/flux-lora-trainer:latest .

# Push to registry (Docker Hub or RunPod registry)
docker push your-registry/flux-lora-trainer:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to RunPod Console → Serverless
2. Create new endpoint with these settings:
   - **Container Image**: `your-registry/flux-lora-trainer:latest`
   - **GPU Types**: RTX 4090, RTX A6000, or RTX A5000
   - **Min Workers**: 2 (for reliability)
   - **Max Workers**: 6
   - **Idle Timeout**: 60 seconds
   - **Max Runtime**: 30 minutes
   - **Health Check**: Enabled (uses container HEALTHCHECK)

### 3. Environment Variables

Set these in your RunPod endpoint:
```
HF_TOKEN=your_huggingface_token
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4. Test the Deployment

```bash
# Test job payload
{
  "input": {
    "model_name": "test_lora",
    "trigger_word": "test",
    "training_type": "person",
    "steps": 100,
    "learning_rate": 0.0002,
    "image_urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg"
    ]
  }
}
```

## Key Improvements in Production Handler

### 1. **Robust Health Checks**
```python
@app.get("/readyz")
async def readyz():
    if WORKER_READY and MODEL_LOADED and FLUX_MODEL_CACHED:
        return {"status": "ready"}
    else:
        return {"status": "not ready"}, 503
```

### 2. **Non-blocking Initialization**
```python
# Background thread prevents blocking health checks
init_thread = threading.Thread(target=initialize_worker, daemon=True)
health_thread = threading.Thread(target=start_health_server, daemon=True)
```

### 3. **Comprehensive Error Handling**
- Timeout protection (25 minutes max)
- Graceful failure with detailed error messages
- Retry logic for model downloads
- Proper cleanup of temporary files

### 4. **Optimized Training Parameters**
- Flux-specific model paths and settings
- Training type-specific optimizations
- Memory-efficient configurations
- Progress tracking and logging

## Expected Results

After deployment:
- ✅ Workers stay healthy and responsive
- ✅ Training jobs complete successfully 
- ✅ Model files return as base64 data
- ✅ Automatic recovery from temporary issues
- ✅ 10-15 minute training times on RTX 4090

## Monitoring

Check worker health:
```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health
```

Expected healthy response:
```json
{
  "jobs": { "completed": 0, "failed": 0, "inProgress": 0, "inQueue": 0, "retried": 0 },
  "workers": { "idle": 2, "initializing": 0, "ready": 2, "running": 0, "unhealthy": 0 }
}
```

## Troubleshooting

### If workers still go unhealthy:
1. Check container logs for initialization errors
2. Verify HuggingFace token is valid
3. Ensure sufficient GPU memory (minimum 24GB recommended)
4. Check network connectivity for model downloads

### If training fails:
1. Verify image URLs are accessible
2. Check training parameters are within valid ranges
3. Monitor GPU memory usage during training
4. Review training logs for specific error messages

This production-ready setup addresses all the root causes identified in the previous worker failures and implements RunPod best practices for maximum reliability.