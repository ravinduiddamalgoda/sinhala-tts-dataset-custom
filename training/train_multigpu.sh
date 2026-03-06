#!/bin/bash
# =============================================================================
# Multi-GPU Training Script (2x RTX 5060 Ti)
# Uses torch distributed data parallel for faster training
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate environment
WORK_DIR="/workspace/sinhala-tts"
source "$WORK_DIR/tts_env/bin/activate"

echo "=========================================="
echo "  Multi-GPU Training (2x GPUs)"
echo "=========================================="

# Check GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Less than 2 GPUs detected. Running single GPU training."
    python "$SCRIPT_DIR/train.py"
else
    echo "Starting distributed training on $GPU_COUNT GPUs..."
    echo ""

    # Start tensorboard in background
    tensorboard --logdir "$SCRIPT_DIR/output" --bind_all --port 6006 &
    echo "TensorBoard running at http://$(hostname -I | awk '{print $1}'):6006"
    echo ""

    # Distributed training via Coqui's trainer
    CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute \
        --script "$SCRIPT_DIR/train.py"
fi
