# Sinhala TTS Training Guide
## Complete Step-by-Step for vast.ai (2x RTX 5060 Ti)

> **Timeline:** Training will produce usable results in 12-24 hours, good quality in 36-48 hours.

---

## Architecture Overview

```
Dataset (6248 clips, 13.7hrs)
├── Female "oshadi": ~985 clips, ~2hrs
└── Male "mettananda": ~5401 clips, ~11.8hrs
         │
         ▼
   VITS Multi-Speaker Model
   (trains on ALL data, both speakers)
         │
         ▼
   Inference with speaker="oshadi"
   → Female voice synthesis
```

**Why multi-speaker?** With only 2 hours of female data, the model benefits from learning shared speech representations from the larger male dataset. The speaker embedding lets you select the female voice at inference time.

---

## Step 1: Provision vast.ai Server

1. Go to [vast.ai](https://vast.ai) → **Create Instance**
2. Filter for:
   - **GPU:** 2x RTX 5060 Ti (or similar with ≥16GB VRAM each)
   - **Disk:** ≥50GB
   - **Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (or similar)
3. Launch and SSH in

---

## Step 2: Setup on the Server

```bash
# SSH into your vast.ai instance
ssh -p <PORT> root@<IP>

# Clone the repo
cd /workspace
git clone https://github.com/pnfo/sinhala-tts-dataset.git
cd sinhala-tts-dataset

# Run the automated setup
bash training/setup_server.sh
```

This installs all dependencies, downloads the audio files, and sets up the Python environment.

### If setup_server.sh fails, do it manually:

```bash
apt-get update && apt-get install -y ffmpeg sox libsox-dev libsndfile1 espeak-ng tmux htop

cd /workspace
mkdir -p sinhala-tts
cd sinhala-tts

git clone https://github.com/pnfo/sinhala-tts-dataset.git
cd sinhala-tts-dataset

# Download audio files
wget "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz"
tar -xzf wavs.tar.gz && rm wavs.tar.gz

# Setup Python
cd /workspace/sinhala-tts
python3 -m venv tts_env
source tts_env/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install TTS==0.22.0 matplotlib tensorboard pandas numpy scipy librosa soundfile tqdm
```

---

## Step 3: Prepare the Dataset

```bash
source /workspace/sinhala-tts/tts_env/bin/activate
cd /workspace/sinhala-tts/sinhala-tts-dataset

python training/prepare_data.py
```

This creates:
- `training/dataset/multispeaker/` — All data, both speakers (recommended)
- `training/dataset/female_only/` — Only oshadi clips
- Corresponding Sinhala-script versions

---

## Step 4: Start Training

### Option A: Multi-GPU (RECOMMENDED — 2x faster)

```bash
# Start in a tmux session so training survives SSH disconnect!
tmux new -s tts

source /workspace/sinhala-tts/tts_env/bin/activate
cd /workspace/sinhala-tts/sinhala-tts-dataset

# Multi-GPU distributed training
CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute \
    --script training/train.py
```

### Option B: Single GPU

```bash
tmux new -s tts
source /workspace/sinhala-tts/tts_env/bin/activate
cd /workspace/sinhala-tts/sinhala-tts-dataset
python training/train.py
```

### Option C: Female-only (single speaker, simpler)

```bash
python training/train_female_only.py
```

---

## Step 5: Monitor Training

### TensorBoard

```bash
# In another tmux pane (Ctrl+B, then %)
source /workspace/sinhala-tts/tts_env/bin/activate
tensorboard --logdir /workspace/sinhala-tts/sinhala-tts-dataset/training/output --bind_all --port 6006
```

Then open `http://<SERVER_IP>:6006` in your browser.

### Check Training Progress

```bash
# See latest loss values
tail -f /workspace/sinhala-tts/sinhala-tts-dataset/training/output/sinhala-vits-multispeaker-*/trainer_0_log.txt
```

### What to look for:
| Step Range | Expected Behavior |
|-----------|-------------------|
| 0-5K | High loss, garbled audio |
| 5K-20K | Loss dropping, some intelligible sounds |
| 20K-50K | Recognizable words, still rough |
| 50K-100K | Good intelligibility, improving naturalness |
| 100K-200K | Good quality, natural prosody |
| 200K+ | Diminishing returns |

**For a 2-day deadline:** You should get usable results by ~50K-100K steps. With 2 GPUs and batch size 32/GPU, this takes approximately 12-24 hours.

---

## Step 6: Inference / Generate Speech

```bash
source /workspace/sinhala-tts/tts_env/bin/activate
cd /workspace/sinhala-tts/sinhala-tts-dataset

# Synthesize a sentence (auto-finds best model)
python training/inference.py \
    --text "ayubōvan, mama ōśadī." \
    --speaker oshadi \
    --output sample_output.wav

# Or specify model explicitly
python training/inference.py \
    --text "apē raṭē kāntāvanṭa dinā gatu yutu ayitivāsikam ræsak tibenavā." \
    --model training/output/sinhala-vits-multispeaker-*/best_model.pth \
    --config training/output/sinhala-vits-multispeaker-*/config.json \
    --speaker oshadi \
    --output sample2.wav
```

---

## Step 7: Demo Interface

```bash
pip install gradio
python training/demo.py
```

Opens a web interface at `http://<SERVER_IP>:7860` (also creates a public share link).

---

## Step 8: Download the Model

```bash
# On the server — compress the model
cd /workspace/sinhala-tts/sinhala-tts-dataset/training/output
tar -czf sinhala-tts-model.tar.gz sinhala-vits-multispeaker-*/best_model.pth sinhala-vits-multispeaker-*/config.json

# On your local machine — download
scp -P <PORT> root@<IP>:/workspace/sinhala-tts/sinhala-tts-dataset/training/output/sinhala-tts-model.tar.gz .
```

---

## Quick Reference: File Structure

```
training/
├── setup_server.sh          # Server setup & dependency installation
├── prepare_data.py          # Dataset preparation
├── train.py                 # Multi-speaker VITS training (RECOMMENDED)
├── train_female_only.py     # Female-only single-speaker training
├── train_multigpu.sh        # Multi-GPU training launcher
├── run_training.sh          # One-click full pipeline
├── inference.py             # Generate speech from trained model
├── demo.py                  # Gradio web demo
├── dataset/                 # Created by prepare_data.py
│   ├── multispeaker/
│   └── female_only/
└── output/                  # Training checkpoints & logs
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce `BATCH_SIZE` in `train.py` from 32 to 16 or 8.

### "No audio files found"
Download manually:
```bash
cd /workspace/sinhala-tts/sinhala-tts-dataset
wget "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz"
tar -xzf wavs.tar.gz
```

### Training is too slow
- Make sure both GPUs are being used: `nvidia-smi`
- Use multi-GPU: `CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script training/train.py`
- Enable mixed precision (already enabled by default)

### Want to resume training after interruption
The trainer auto-saves checkpoints. It will automatically resume from the latest checkpoint when you restart training.

### "TTS module not found"
```bash
source /workspace/sinhala-tts/tts_env/bin/activate
pip install TTS==0.22.0
```

---

## Expected Timeline (2x RTX 5060 Ti)

| Time | Steps (~) | Quality |
|------|-----------|---------|
| 2 hrs | ~5K | Garbled |
| 6 hrs | ~15K | Some words |
| 12 hrs | ~35K | Intelligible |
| 24 hrs | ~70K | Good for demo |
| 36 hrs | ~100K | Good quality |
| 48 hrs | ~140K | Best within timeframe |

**For your 2-day demo:** Start training ASAP. By hour 24 you'll have a demo-ready model.
