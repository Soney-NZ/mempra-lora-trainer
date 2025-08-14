#!/usr/bin/env python3
"""
Test script for Flux LoRA training endpoint
Based on RunPod SDXL tutorial patterns
"""

import requests
import json
import time
import base64
import os
from pathlib import Path

# Configuration
ENDPOINT_ID = "k7izyh2payn34x"  # Your RunPod endpoint ID
API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

if not API_KEY:
    print("ERROR: RUNPOD_API_KEY environment variable not set")
    exit(1)

def submit_training_job():
    """Submit a Flux LoRA training job (following SDXL pattern)"""
    
    # Sample training job payload
    payload = {
        "input": {
            "model_name": "test_flux_lora",
            "training_type": "person",
            "trigger_word": "ohwx",
            "steps": 100,  # Reduced for testing
            "learning_rate": 0.0002,
            "batch_size": 1,
            "image_urls": [
                "https://example.com/test1.jpg",
                "https://example.com/test2.jpg",
                "https://example.com/test3.jpg"
            ]
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    print("Submitting Flux LoRA training job...")
    print(f"Endpoint: {BASE_URL}/run")
    
    response = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get('id')
        print(f"✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"Status: {result.get('status')}")
        return job_id
    else:
        print(f"❌ Job submission failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_job_status(job_id):
    """Check job status (following SDXL pattern)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Status check failed: {response.status_code}")
        return None

def monitor_job(job_id):
    """Monitor job until completion (following SDXL pattern)"""
    print(f"\nMonitoring job {job_id}...")
    
    while True:
        status_data = check_job_status(job_id)
        
        if not status_data:
            break
            
        status = status_data.get('status')
        print(f"Status: {status}")
        
        if status == 'COMPLETED':
            print("✅ Job completed successfully!")
            return status_data
            
        elif status == 'FAILED':
            print("❌ Job failed!")
            print(f"Error: {status_data.get('error', 'Unknown error')}")
            return status_data
            
        elif status in ['IN_QUEUE', 'IN_PROGRESS']:
            delay_time = status_data.get('delayTime', 0)
            execution_time = status_data.get('executionTime', 0)
            
            if delay_time > 0:
                print(f"  Queue delay: {delay_time/1000:.1f}s")
            if execution_time > 0:
                print(f"  Execution time: {execution_time/1000:.1f}s")
                
            print("  Waiting 10 seconds before next check...")
            time.sleep(10)
        else:
            print(f"  Unknown status: {status}")
            time.sleep(5)

def save_model_from_result(result, filename="test_flux_lora.safetensors"):
    """Save the trained model from base64 data (following SDXL pattern)"""
    
    output = result.get('output', {})
    model_b64 = output.get('model_data_b64')
    
    if not model_b64:
        print("❌ No model data found in result")
        return False
    
    try:
        # Decode base64 model data
        model_data = base64.b64decode(model_b64)
        
        # Save to file
        with open(filename, 'wb') as f:
            f.write(model_data)
            
        file_size = len(model_data)
        print(f"✅ Model saved as '{filename}' ({file_size:,} bytes)")
        print(f"Model path: {os.path.abspath(filename)}")
        
        # Display model info
        model_name = output.get('model_name', 'unknown')
        trigger_word = output.get('trigger_word', 'unknown')
        steps = output.get('steps_completed', 'unknown')
        
        print(f"Model details:")
        print(f"  Name: {model_name}")
        print(f"  Trigger word: {trigger_word}")
        print(f"  Training steps: {steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def check_endpoint_health():
    """Check endpoint health before testing"""
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    print("Checking endpoint health...")
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    
    if response.status_code == 200:
        health_data = response.json()
        workers = health_data.get('workers', {})
        
        ready = workers.get('ready', 0)
        idle = workers.get('idle', 0)
        unhealthy = workers.get('unhealthy', 0)
        initializing = workers.get('initializing', 0)
        
        print(f"Endpoint health:")
        print(f"  Ready workers: {ready}")
        print(f"  Idle workers: {idle}")  
        print(f"  Unhealthy workers: {unhealthy}")
        print(f"  Initializing workers: {initializing}")
        
        active_workers = ready + idle
        if active_workers > 0:
            print("✅ Endpoint is healthy and ready for jobs")
            return True
        else:
            print("⚠️ No active workers available")
            return False
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Flux LoRA RunPod Endpoint")
    print("=" * 50)
    
    # Step 1: Check endpoint health
    if not check_endpoint_health():
        print("\n❌ Endpoint not ready - please wait for workers to initialize")
        return
    
    # Step 2: Submit training job
    job_id = submit_training_job()
    if not job_id:
        return
    
    # Step 3: Monitor job progress  
    result = monitor_job(job_id)
    if not result:
        return
    
    # Step 4: Save trained model if successful
    if result.get('status') == 'COMPLETED':
        save_model_from_result(result)
        print("\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed with status: {result.get('status')}")

if __name__ == "__main__":
    main()