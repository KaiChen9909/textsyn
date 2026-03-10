#!/bin/bash

# ==========================================
# Download Pretrained CTCL-Topic Model
# ==========================================
# This script downloads the pretrained CTCL-Topic model
# trained on Wikipedia with ~1K topics.

# ==========================================
# Configuration
# ==========================================
MODEL_DIR="./models"
GDRIVE_ID="1sbda6ROyMewThuoDA3bxP71ucihcf7qJ"
MODEL_NAME="ctcl_topic"

# ==========================================
# Download
# ==========================================
echo "=== Downloading CTCL Pretrained Models ==="
echo "Model will be saved to: ${MODEL_DIR}"
echo ""

# Create model directory
mkdir -p ${MODEL_DIR}
cd ${MODEL_DIR}

# Install gdown if not available
pip install -q gdown

# Download from Google Drive
echo "Downloading from Google Drive..."
gdown ${GDRIVE_ID}

# Unzip
echo "Extracting model files..."
unzip -q ctcl_pretrained.zip

echo ""
echo "=== Download Complete ==="
echo "CTCL-Topic model location: ${MODEL_DIR}/${MODEL_NAME}"
echo ""
echo "You can now use this model for topic extraction:"
echo "  python extract_topics.py --model_path ${MODEL_DIR}/${MODEL_NAME} ..."
