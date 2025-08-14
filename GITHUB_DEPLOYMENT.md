# GitHub Deployment for RunPod LoRA Training

## Quick Setup (2 minutes)

### 1. GitHub Repository Setup
```bash
# Add runpod-setup to your GitHub repo
git add runpod-setup/
git commit -m "Add production RunPod LoRA handler"
git push origin main
```

### 2. GitHub Secrets Configuration
Go to your GitHub repo → Settings → Secrets and variables → Actions

Add these secrets:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password (or access token)

### 3. Deploy Workflow
```bash
# Copy the workflow file
mkdir -p .github/workflows/
cp runpod-setup/github-deploy.yml .github/workflows/

# Commit and push
git add .github/workflows/github-deploy.yml
git commit -m "Add RunPod deployment workflow"
git push origin main
```

### 4. Trigger Deployment
The workflow will automatically run when you push changes to `runpod-setup/` or you can trigger it manually:
1. Go to your GitHub repo → Actions
2. Select "Deploy Flux LoRA Handler to RunPod" 
3. Click "Run workflow"

## After Deployment

### Update RunPod Endpoint
1. Go to RunPod Console → Serverless
2. Edit your existing endpoint or create new one
3. Update container image to: `your-username/flux-lora-trainer:latest`
4. Configure settings:
   - **GPU Types**: RTX 4090, RTX A6000, RTX A5000
   - **Min Workers**: 2
   - **Max Workers**: 6
   - **Idle Timeout**: 60 seconds
   - **Max Runtime**: 30 minutes

### Environment Variables
Set in RunPod endpoint:
```
HF_TOKEN=your_huggingface_token
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Test Deployment
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model_name": "test_lora",
      "trigger_word": "test",
      "training_type": "person",
      "steps": 100,
      "learning_rate": 0.0002,
      "image_urls": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512",
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=512"
      ]
    }
  }'
```

## Benefits of GitHub Deployment

✅ **Automated**: Push code → Auto-build → Auto-deploy
✅ **Versioned**: Every deployment is tagged with commit hash
✅ **Rollback**: Easy to revert to previous working version
✅ **CI/CD**: Integrated testing and validation
✅ **Team-friendly**: Multiple developers can contribute safely

## File Structure for GitHub

```
your-repo/
├── runpod-setup/
│   ├── handler-production.py       # Production handler
│   ├── Dockerfile.production       # Optimized container
│   ├── requirements.txt           # Dependencies
│   ├── DEPLOYMENT_GUIDE.md        # Deployment docs
│   └── README-PRODUCTION.md       # Production setup
├── .github/
│   └── workflows/
│       └── github-deploy.yml      # Deployment workflow
└── replit.md                     # Project documentation
```

This setup gives you professional-grade deployment automation for your RunPod LoRA training system!