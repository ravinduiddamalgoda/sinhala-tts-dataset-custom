#!/bin/bash
# ================================================================================
# SINHALA TTS - IMPROVED FEMALE VOICE TRAINING (vast.ai)
# ================================================================================
# Trains a single-speaker VITS model on preprocessed oshadi (female) data.
# Improvements over the original multi-speaker approach:
#   - Audio loudness normalization (LUFS) + silence trimming
#   - Optional denoising (spectral gating)
#   - mel_fmax=8000 (speech-only frequencies)
#   - Tuned hyperparameters for small dataset (lower LR, add_blank, FP32)
#   - Inference post-processing (highpass + loudness normalization)
#
# Usage: Copy this entire script and paste into your vast.ai terminal
# ================================================================================
set -e

echo "=========================================="
echo "  Sinhala TTS - Improved Female Voice"
echo "=========================================="

# ---- STEP 1: System dependencies + Python 3.11 ----
echo ""
echo "[1/7] Installing system dependencies + Python 3.11..."
apt-get update -qq && apt-get install -y -qq \
    software-properties-common 2>/dev/null
add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null
apt-get update -qq && apt-get install -y -qq \
    git wget ffmpeg sox libsox-dev \
    python3.11 python3.11-venv python3.11-dev \
    build-essential cmake \
    libsndfile1 espeak-ng \
    tmux htop nvtop 2>/dev/null

# ---- STEP 2: Verify SCP'd dataset ----
PROJECT_DIR="/workspace/sinhala-tts-dataset-custom"
echo ""
echo "[2/7] Verifying dataset at $PROJECT_DIR..."
cd "$PROJECT_DIR"

if [ ! -f "metadata.csv" ]; then
    echo "ERROR: metadata.csv not found at $PROJECT_DIR"
    echo "SCP it from your local machine first."
    exit 1
fi

# Check wavs - need oshadi files (sinh_5402+) from the full v2.1 dataset
WAV_COUNT=$(ls wavs/*.wav 2>/dev/null | wc -l)
echo "  Found $WAV_COUNT wav files"

if [ ! -d "wavs" ] || [ "$WAV_COUNT" -lt 100 ]; then
    echo "ERROR: wavs/ directory missing or incomplete."
    echo "SCP the wavs folder from your local machine first."
    exit 1
fi

# Check if oshadi files exist (they start at sinh_5402)
OSHADI_COUNT=$(ls wavs/sinh_5[4-9]*.wav wavs/sinh_6*.wav 2>/dev/null | wc -l)
if [ "$OSHADI_COUNT" -lt 100 ]; then
    echo ""
    echo "WARNING: Only $OSHADI_COUNT oshadi (female) wav files found."
    echo "The 'old dataset/wavs/' only has male speaker files."
    echo "You need the FULL v2.1 wavs with sinh_5402 - sinh_6385 (oshadi clips)."
    echo ""
    echo "SCP the correct wavs from your local machine:"
    echo "  scp -P 30325 -r \"c:/Users/HP/Documents/SLIIT/Research/sinhala-tts-dataset/wavs\" root@12.16.140.162:/workspace/sinhala-tts-dataset-custom/"
    exit 1
fi
echo "  Found $OSHADI_COUNT oshadi (female) wav files"

# ---- STEP 3: Verify training scripts ----
echo ""
echo "[3/7] Verifying training scripts..."
if [ ! -f "training/preprocess_audio.py" ]; then
    echo "ERROR: training scripts not found."
    echo "SCP the training/ folder from your local machine first."
    exit 1
fi
echo "  Training scripts OK."

# ---- STEP 4: Python 3.11 venv + dependencies ----
echo ""
echo "[4/7] Setting up Python 3.11 virtual environment..."
VENV_DIR="/workspace/tts_env"
if [ -d "$VENV_DIR" ]; then
    # Remove old venv if it was created with wrong Python version
    OLD_PY_VERSION=$("$VENV_DIR/bin/python" --version 2>/dev/null || echo "unknown")
    if [[ "$OLD_PY_VERSION" != *"3.11"* ]]; then
        echo "  Removing old venv ($OLD_PY_VERSION), need Python 3.11..."
        rm -rf "$VENV_DIR"
    fi
fi
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "  Python version: $(python --version)"
pip install --upgrade pip setuptools wheel -q

# PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Coqui TTS (requires Python <3.12)
pip install TTS==0.22.0 -q

# Core dependencies
pip install matplotlib tensorboard pandas numpy scipy librosa soundfile Unidecode inflect tqdm -q

# Audio preprocessing dependencies
pip install pyloudnorm noisereduce -q

echo "  Python environment ready."

# ---- STEP 5: Preprocess audio ----
echo ""
echo "[5/7] Preprocessing oshadi audio (loudness normalization + trimming)..."
cd "$PROJECT_DIR"
python training/preprocess_audio.py --speaker oshadi
echo ""
echo "  Check preprocessing_report.txt for details."

# ---- STEP 6: Prepare cleaned dataset ----
echo ""
echo "[6/7] Preparing cleaned dataset..."
python training/prepare_data.py

# ---- STEP 7: Summary ----
echo ""
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "  To start training:"
echo ""
echo "    tmux new -s tts"
echo "    source /workspace/tts_env/bin/activate"
echo "    cd /workspace/sinhala-tts-dataset-custom"
echo ""
echo "    # Single GPU:"
echo "    python training/train_female_only.py"
echo ""
echo "    # Monitor (in another tmux pane: Ctrl+B then %):"
echo "    tensorboard --logdir training/output_female_only_clean --bind_all --port 6006"
echo ""
echo "    # After training, test inference:"
echo "    python training/inference.py --text \"ayubovan, mama ōśadī.\""
echo ""
echo "  TIPS:"
echo "    - Training should produce usable results after ~50K steps"
echo "    - Listen to checkpoints at different steps to find best quality"
echo "    - If voice is too robotic, try re-running with --denoise:"
echo "      python training/preprocess_audio.py --speaker oshadi --denoise"
echo "      python training/prepare_data.py"
echo ""
