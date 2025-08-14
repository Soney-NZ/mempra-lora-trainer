from fastapi import FastAPI
import os, psutil, subprocess

app = FastAPI()
READY_FLAG = "/tmp/ready"

@app.get("/readyz")
def readyz():
    return {"status": "ready" if os.path.exists(READY_FLAG) else "initializing"}

@app.get("/livez")
def livez():
    return {"status": "alive"}

@app.get("/status")
def status():
    gpu_info = "nvidia-smi not available"
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True)
    except Exception:
        pass
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "mem_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "gpu_info": gpu_info.strip()
    }