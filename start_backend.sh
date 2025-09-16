#!/bin/bash

# Start the ISL Prediction Backend Server
echo "Starting ISL Prediction Backend Server..."

# Change to the INCLUDE-ISL directory
cd INCLUDE-ISL

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install fastapi uvicorn

# Check if model file exists
MODEL_PATH="../include_bert/include/bert/epoch=328-step=18094.ckpt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the BERT model is properly placed in the include_bert directory"
    exit 1
fi

# Start the server
echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
python backend.py
