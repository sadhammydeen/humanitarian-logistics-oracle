#!/bin/bash
echo "Installing project requirements..."
pip install -r requirements.txt || true

echo "Starting FastAPI tracking server on port 8000..."
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
