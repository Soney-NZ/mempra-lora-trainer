#!/bin/bash
# Production deployment script for Flux LoRA training on RunPod
# Use this for local deployment or GitHub Actions for automated deployment

set -e

echo "🚀 Deploying Flux LoRA Training Handler to RunPod..."

# Configuration - Update with your Docker Hub username
DOCKER_USERNAME="${1:-your-docker-username}"  # Pass as argument or edit this
IMAGE_NAME="flux-lora-trainer"
VERSION="v1.0.0"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

if [ "$DOCKER_USERNAME" = "your-docker-username" ]; then
    echo "❌ Please provide your Docker Hub username:"
    echo "   ./deploy-production.sh your-username"
    echo "   or edit the script to set DOCKER_USERNAME"
    exit 1
fi

# Check required files exist
echo "📋 Checking deployment files..."
required_files=(
    "handler-production.py"
    "Dockerfile.production" 
    "requirements.txt"
    "DEPLOYMENT_GUIDE.md"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

echo "✅ All deployment files present"

# Validate handler syntax
echo "🔍 Validating handler syntax..."
python -m py_compile handler-production.py
echo "✅ Handler syntax valid"

# Build Docker image
echo "🔨 Building Docker image: $FULL_IMAGE"
docker build -f Dockerfile.production -t "$FULL_IMAGE" .

# Test image locally (optional)
echo "🧪 Testing image locally..."
docker run --rm -e RUNPOD_API_KEY=test "$FULL_IMAGE" python -c "
import handler_production
print('✅ Handler imports successfully')
"

# Push to registry
echo "📤 Pushing image to registry..."
docker push "$FULL_IMAGE"

echo "🎉 Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Go to RunPod Console → Serverless" 
echo "2. Create endpoint with image: $FULL_IMAGE"
echo "3. Set GPU types: RTX 4090, RTX A6000"
echo "4. Configure Min Workers: 2, Max Workers: 6"
echo "5. Set Environment Variables:"
echo "   - HF_TOKEN=your_huggingface_token"
echo "   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "6. Enable health checks"
echo ""
echo "🔗 Update RUNPOD_LORA_ENDPOINT_ID in your Mempra environment"
echo "   with the new endpoint ID once deployed"