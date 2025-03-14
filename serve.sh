#!/bin/bash
set -e

# Create logs directory (ensure proper permissions)
mkdir -p /home/model-server/logs

# Check if whisper model exists and report
MODEL_DIR="/home/model-server/whisper-models"
MODEL_FILE="$MODEL_DIR/whisper_base.pt"

if [ -f "$MODEL_FILE" ] && [ -s "$MODEL_FILE" ]; then
    echo "✓ Found pre-downloaded model at $MODEL_FILE"
else
    echo "! Model file not found or empty. Will download during initialization (this may take time and duplicate downloads)"
fi

# Create model archive
echo "Creating model archive..."
/home/model-server/model-archiver/create_model_archive.sh

# Verify the model archive exists
echo "Checking model archive..."
ls -la /home/model-server/model-store/
if [ ! -f "/home/model-server/model-store/whisper.mar" ]; then
    echo "ERROR: whisper.mar not found in model store!"
    exit 1
fi

# Stop any running TorchServe instance (if any) and wait a bit
torchserve --stop || echo "No previous TorchServe instance running"
sleep 5

# Start TorchServe using minimal configuration - no log-config
echo "Starting TorchServe..."
torchserve --start \
  --ts-config /home/model-server/config/config.properties \
  --model-store /home/model-server/model-store \
  --models whisper.mar \
  --disable-token-auth \
  --ncs

echo "Waiting for TorchServe to initialize..."
sleep 15

# Keep container running by tailing the logs
tail -f /home/model-server/logs/*.log