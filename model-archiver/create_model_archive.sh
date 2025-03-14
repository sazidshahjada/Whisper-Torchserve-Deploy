#!/bin/bash
set -e

echo "Installing model requirements..."
pip install --no-cache-dir -r /home/model-server/model-archiver/requirements.txt

echo "Creating model archive..."
# Check if handler directory exists
if [ ! -d "/home/model-server/model-archiver/handler" ]; then
    echo "ERROR: Handler directory not found!"
    exit 1
fi

# List handlers for debugging
echo "Available handlers:"
ls -la /home/model-server/model-archiver/handler/

# Check if the main handler file exists
if [ ! -f "/home/model-server/model-archiver/handler/whisper_handler.py" ]; then
    echo "ERROR: whisper_handler.py not found!"
    exit 1
fi

# Create model archive without extra-files parameter since whisper_utils.py might not exist
torch-model-archiver --model-name whisper \
  --version 1.0 \
  --handler /home/model-server/model-archiver/handler/whisper_handler.py \
  --runtime python \
  --export-path /home/model-server/model-store

echo "Model archive created successfully at /home/model-server/model-store/whisper.mar"

# Verify the archive was created
ls -la /home/model-server/model-store/