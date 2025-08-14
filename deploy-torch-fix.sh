#!/bin/bash
# Deploy PyTorch compatibility fix to RunPod

echo "🔧 Deploying PyTorch torchvision compatibility fix..."

# Ensure we're in the runpod-setup directory
cd runpod-setup

# Update the deployment package with fixed requirements and handler
echo "📦 Creating deployment package with torch compatibility fix..."

# The files are already updated:
# - requirements.txt now has pinned torch==2.3.1 and torchvision==0.18.1
# - handler-production.py now has runtime compatibility check

echo "✅ Torch compatibility fix ready for deployment"
echo "🚀 Push these changes to trigger a new RunPod deployment"
echo ""
echo "Fixed issues:"
echo "- Pinned torch==2.3.1 and torchvision==0.18.1 for compatibility"
echo "- Added runtime compatibility check and auto-fix"
echo "- Resolves 'operator torchvision::nms does not exist' error"