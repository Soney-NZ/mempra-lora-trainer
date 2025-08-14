#!/bin/bash
set -e
uvicorn healthcheck:app --host 0.0.0.0 --port 8000 &
python -u handler.py