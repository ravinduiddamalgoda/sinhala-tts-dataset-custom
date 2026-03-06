#!/bin/bash
# =============================================================================
# Quick-Start: Run everything in one go on the vast.ai server
# =============================================================================
# Usage: bash training/run_training.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="/workspace/sinhala-tts"

echo "=========================================="
echo "  Sinhala TTS - Full Training Pipeline"
echo "=========================================="

# --- Step 1: Setup (if not done) ---
if [ ! -d "$WORK_DIR/tts_env" ]; then
    echo "[Step 1] Running server setup..."
    bash "$SCRIPT_DIR/setup_server.sh"
else
    echo "[Step 1] Environment already exists, skipping setup."
fi

# Activate environment
source "$WORK_DIR/tts_env/bin/activate"

# --- Step 2: Verify dataset ---
WAVS_DIR="$PROJECT_DIR/wavs"
if [ ! -d "$WAVS_DIR" ] || [ -z "$(ls -A "$WAVS_DIR" 2>/dev/null)" ]; then
    echo "[Step 2] Downloading audio files..."
    cd "$PROJECT_DIR"
    wget -q --show-progress "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz" -O wavs.tar.gz
    tar -xzf wavs.tar.gz
    rm wavs.tar.gz
fi

WAV_COUNT=$(ls "$WAVS_DIR"/*.wav 2>/dev/null | wc -l)
echo "[Step 2] Found $WAV_COUNT wav files"

# --- Step 3: Prepare data ---
echo "[Step 3] Preparing dataset..."
python "$SCRIPT_DIR/prepare_data.py"

# --- Step 4: Train ---
echo "[Step 4] Starting VITS training..."
echo ""
echo "IMPORTANT: Training will take 24-48 hours."
echo "Monitor with: tensorboard --logdir $SCRIPT_DIR/output --bind_all"
echo ""
echo "For multi-GPU (2x GPUs), run instead:"
echo "  CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script $SCRIPT_DIR/train.py"
echo ""

# Single GPU training (simpler, still fast)
python "$SCRIPT_DIR/train.py"
