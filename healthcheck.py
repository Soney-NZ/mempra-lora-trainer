#!/usr/bin/env python3
"""
RunPod Health Contract Implementation
Follows RunPod's official health check requirements for proper worker management
"""

import os
import time
from fastapi import FastAPI, Response
import logging

logger = logging.getLogger(__name__)

# Health status flags
READY_FLAG = "/tmp/ready"
INITIALIZING_FLAG = "/tmp/initializing"

app = FastAPI(title="RunPod Health Service", version="1.0.0")

@app.get("/readyz")  # Keep for Kubernetes compatibility
def readyz():
    """Kubernetes-style readiness probe"""
    if os.path.exists(READY_FLAG):
        return {"status": "ready", "timestamp": time.time()}
    else:
        return Response(
            content='{"status": "initializing", "timestamp": %d}' % time.time(),
            status_code=503,
            media_type="application/json"
        )

@app.get("/ping")  # RunPod expects this specific endpoint
def ping():
    """
    RunPod health contract endpoint
    - Returns 200 when worker is ready to accept jobs
    - Returns 204 when worker is initializing (removes from load balancer)
    - Returns non-2xx when unhealthy (worker gets terminated)
    """
    if os.path.exists(READY_FLAG):
        # 200 = healthy/ready - worker can accept jobs
        return {"ok": True, "status": "ready", "timestamp": time.time()}
    elif os.path.exists(INITIALIZING_FLAG):
        # 204 = initializing - keep worker alive but don't send jobs
        return Response(status_code=204)
    else:
        # 503 = not ready yet - keep worker alive but don't send jobs
        return Response(
            content='{"status": "starting", "timestamp": %d}' % time.time(),
            status_code=503,
            media_type="application/json"
        )

@app.get("/health") 
def health():
    """Basic liveness check - container is running"""
    return {
        "status": "alive",
        "service": "flux-lora-trainer",
        "version": "2.0.0",
        "timestamp": time.time(),
        "ready": os.path.exists(READY_FLAG),
        "initializing": os.path.exists(INITIALIZING_FLAG)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_HEALTH", 8000))
    logger.info(f"[Health] Starting RunPod health service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")