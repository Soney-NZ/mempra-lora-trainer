# GitHub Repository Setup for RunPod

## Step-by-Step Instructions:

### 1. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `mempra-lora-trainer`
3. Description: "LoRA training endpoint for Mempra AI"
4. Make it Public (RunPod needs access)
5. Click "Create repository"

### 2. Upload Files
Upload these 4 files to your new repository:
- `handler.py` (main training logic)
- `requirements.txt` (Python dependencies)  
- `Dockerfile` (container setup)
- `README.md` (documentation)

You can either:
- Use GitHub's web interface to upload files
- Or clone the repo and push files via command line

### 3. Get Repository URL
Copy your repository URL (format: `https://github.com/yourusername/mempra-lora-trainer`)

### 4. Create RunPod Endpoint
1. Go to https://console.runpod.io/serverless/new-endpoint
2. Click "Import Git Repository"
3. Paste your GitHub URL
4. Settings:
   - Name: `mempra-lora-trainer`
   - GPU Type: RTX A6000 (recommended)
   - Container Disk: 20GB
   - Max Workers: 2
   - Python Version: 3.10
5. Click "Deploy"

### 5. Get Endpoint ID
After deployment (takes 3-5 minutes):
- Copy the endpoint ID (format: `abc123-def456-789`)
- Add to Replit Secrets: `RUNPOD_LORA_ENDPOINT_ID`

### 6. Test Training
Your system will automatically use RunPod for real GPU training!

## Benefits:
- ✅ Real Kohya training (industry standard)
- ✅ Flux 1.1 base model  
- ✅ RTX A6000 GPU acceleration
- ✅ Professional quality LoRA models
- ✅ Cost-effective pay-per-use pricing
- ✅ No Docker setup required