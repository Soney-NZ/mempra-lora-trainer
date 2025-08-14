#!/usr/bin/env python3
"""
Test script to validate the production handler locally
"""

import asyncio
import json

# Mock job input for testing
test_job = {
    "id": "test-job-123",
    "input": {
        "model_name": "test_lora",
        "trigger_word": "ohwx",
        "training_type": "person", 
        "steps": 50,  # Small number for testing
        "learning_rate": 0.0002,
        "image_urls": [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop",
            "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=512&h=512&fit=crop"
        ]
    }
}

async def test_handler():
    try:
        # Import handler
        from handler_production import handler
        
        print("🧪 Testing production handler...")
        print(f"📋 Test job: {json.dumps(test_job, indent=2)}")
        
        # Run handler
        result = await handler(test_job)
        
        print("✅ Handler completed successfully!")
        print(f"📊 Result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"❌ Handler test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_handler())