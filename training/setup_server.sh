#!/bin/bash
# =============================================================================
# Sinhala TTS - Server Setup Script for vast.ai (2x RTX 5060 Ti)
# Run this FIRST on your vast.ai instance
# =============================================================================
set -e

echo "=========================================="
echo "  Sinhala TTS Training - Server Setup"
echo "=========================================="

# Update system
apt-get update && apt-get install -y \
    git wget ffmpeg sox libsox-dev \
    python3-pip python3-venv \
    build-essential cmake \
    libsndfile1 \
    espeak-ng \
    tmux htop nvtop

# Create workspace
WORK_DIR="/workspace/sinhala-tts"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone the dataset
if [ ! -d "sinhala-tts-dataset" ]; then
    echo "Cloning dataset repository..."
    git clone https://github.com/pnfo/sinhala-tts-dataset.git
fi

# Download the wavs from the release
cd sinhala-tts-dataset
if [ ! -d "wavs" ] || [ -z "$(ls -A wavs 2>/dev/null)" ]; then
    echo "Downloading audio files from release..."
    # v2.1 release - download the tar file with wavs
    wget -q --show-progress "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz" -O wavs.tar.gz
    tar -xzf wavs.tar.gz
    rm wavs.tar.gz
    echo "Audio files extracted to wavs/"
fi

# Verify wavs exist
WAV_COUNT=$(ls wavs/*.wav 2>/dev/null | wc -l)
echo "Found $WAV_COUNT wav files"
if [ "$WAV_COUNT" -lt 1000 ]; then
    echo "WARNING: Expected ~6248 wav files but found $WAV_COUNT"
    echo "You may need to manually download from:"
    echo "https://github.com/pnfo/sinhala-tts-dataset/releases"
fi

cd "$WORK_DIR"

# Create Python virtual environment
echo "Setting up Python environment..."
python3 -m venv tts_env
source tts_env/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Coqui TTS
pip install TTS==0.22.0

# Install additional dependencies
pip install \
    matplotlib \
    tensorboard \
    pandas \
    numpy \
    scipy \
    librosa \
    soundfile \
    Unidecode \
    inflect \
    tqdm

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd $WORK_DIR"
echo "  2. source tts_env/bin/activate"
echo "  3. python sinhala-tts-dataset/training/prepare_data.py"
echo "  4. python sinhala-tts-dataset/training/train.py"
echo ""
