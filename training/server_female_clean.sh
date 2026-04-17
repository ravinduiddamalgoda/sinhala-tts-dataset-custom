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

# ---- STEP 1: System dependencies ----
echo ""
echo "[1/8] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
    git wget ffmpeg sox libsox-dev \
    python3-pip python3-venv \
    build-essential cmake \
    libsndfile1 espeak-ng \
    tmux htop nvtop 2>/dev/null

# ---- STEP 2: Clone dataset repo ----
WORK_DIR="/workspace/sinhala-tts"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "sinhala-tts-dataset" ]; then
    echo ""
    echo "[2/8] Cloning dataset repository..."
    git clone https://github.com/pnfo/sinhala-tts-dataset.git
else
    echo ""
    echo "[2/8] Dataset repo already exists."
fi

PROJECT_DIR="$WORK_DIR/sinhala-tts-dataset"
cd "$PROJECT_DIR"

# ---- STEP 3: Download audio files from GitHub Releases ----
if [ ! -d "wavs" ] || [ "$(ls wavs/*.wav 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo ""
    echo "[3/8] Downloading audio files (~1.5GB)..."
    wget --show-progress "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz" -O wavs.tar.gz
    tar -xzf wavs.tar.gz
    rm -f wavs.tar.gz
else
    echo ""
    echo "[3/8] Audio files already present."
fi
echo "  Found $(ls wavs/*.wav 2>/dev/null | wc -l) wav files"

# ---- STEP 4: Python venv + dependencies ----
echo ""
echo "[4/8] Setting up Python virtual environment..."
cd "$WORK_DIR"
if [ ! -d "tts_env" ]; then
    python3 -m venv tts_env
fi
source tts_env/bin/activate

pip install --upgrade pip setuptools wheel -q

# PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Coqui TTS
pip install TTS==0.22.0 -q

# Core dependencies
pip install matplotlib tensorboard pandas numpy scipy librosa soundfile Unidecode inflect tqdm -q

# Audio preprocessing dependencies (NEW)
pip install pyloudnorm noisereduce -q

echo "  Python environment ready."

# ---- STEP 5: Copy improved training scripts ----
echo ""
echo "[5/8] Creating improved training scripts..."
TRAIN_DIR="$PROJECT_DIR/training"
mkdir -p "$TRAIN_DIR"

# ===== preprocess_audio.py (NEW) =====
cat > "$TRAIN_DIR/preprocess_audio.py" << 'PYTHON_PREPROCESS_EOF'
"""
Sinhala TTS - Audio Preprocessing Pipeline
===========================================
Processes WAV files for higher quality training:
  - Loudness normalization (LUFS)
  - Silence trimming
  - Optional denoising
  - SNR estimation & quality filtering
  - Duration and RMS filtering

Usage:
  python preprocess_audio.py                        # process oshadi (female) clips
  python preprocess_audio.py --speaker all          # process all speakers
  python preprocess_audio.py --denoise              # enable spectral denoising
  python preprocess_audio.py --denoise --prop 0.6   # stronger denoising
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

try:
    import pyloudnorm as pyln
except ImportError:
    print("ERROR: pyloudnorm not installed. Run: pip install pyloudnorm")
    sys.exit(1)

# Configuration
DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")
WAVS_DIR = os.path.join(DATASET_ROOT, "wavs")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "wavs_processed")
SAMPLE_RATE = 22050

# Quality thresholds
TARGET_LUFS = -23.0
MIN_DURATION = 1.5
MAX_DURATION = 12.0
MIN_RMS = 0.01
MIN_SNR_DB = 15.0
TRIM_TOP_DB = 25


def estimate_snr(audio, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.mean(frames ** 2, axis=0)
    if len(frame_energy) == 0:
        return 0.0
    noise_floor = np.percentile(frame_energy, 20)
    signal_energy = np.mean(frame_energy)
    if noise_floor <= 0:
        return 60.0
    snr = 10 * np.log10(signal_energy / noise_floor)
    return float(snr)


def process_single_file(args_tuple):
    wav_id, wav_path, output_path, do_denoise, denoise_prop = args_tuple
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        audio_trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
        duration = len(audio_trimmed) / sr
        if duration < MIN_DURATION:
            return (wav_id, "rejected", f"too_short: {duration:.2f}s < {MIN_DURATION}s")
        if duration > MAX_DURATION:
            return (wav_id, "rejected", f"too_long: {duration:.2f}s > {MAX_DURATION}s")

        rms = np.sqrt(np.mean(audio_trimmed ** 2))
        if rms < MIN_RMS:
            return (wav_id, "rejected", f"too_quiet: RMS={rms:.4f} < {MIN_RMS}")

        snr = estimate_snr(audio_trimmed, sr)
        snr_warning = ""
        if snr < MIN_SNR_DB:
            snr_warning = f" [LOW_SNR: {snr:.1f}dB]"

        if do_denoise:
            import noisereduce as nr
            audio_trimmed = nr.reduce_noise(
                y=audio_trimmed, sr=sr, prop_decrease=denoise_prop, stationary=True,
            )

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_trimmed)
        if loudness < -70.0:
            return (wav_id, "rejected", f"silence: loudness={loudness:.1f} LUFS")
        audio_normalized = pyln.normalize.loudness(audio_trimmed, loudness, TARGET_LUFS)
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        sf.write(output_path, audio_normalized, sr, subtype="PCM_16")
        final_duration = len(audio_normalized) / sr
        return (wav_id, "ok", f"dur={final_duration:.2f}s rms={rms:.3f} snr={snr:.1f}dB{snr_warning}")
    except Exception as e:
        return (wav_id, "error", str(e))


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for TTS training")
    parser.add_argument("--speaker", type=str, default="oshadi",
                        help="Speaker: 'oshadi', 'mettananda', or 'all' (default: oshadi)")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable spectral denoising (conservative)")
    parser.add_argument("--prop", type=float, default=0.5,
                        help="Denoising strength 0.0-1.0 (default: 0.5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Audio Preprocessing")
    print("=" * 60)
    print(f"  Speaker:    {args.speaker}")
    print(f"  Denoise:    {args.denoise} (prop={args.prop})")
    print(f"  LUFS:       {TARGET_LUFS}")
    print(f"  Duration:   {MIN_DURATION}s - {MAX_DURATION}s")

    if not os.path.exists(METADATA_FILE):
        print(f"\nERROR: metadata.csv not found at {METADATA_FILE}")
        sys.exit(1)

    entries = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) < 4: continue
            wav_id, roman, sinhala, speaker = parts[0], parts[1], parts[2], parts[3]
            if args.speaker != "all" and speaker != args.speaker:
                continue
            entries.append({"wav_id": wav_id, "roman": roman, "sinhala": sinhala, "speaker": speaker})

    print(f"  Entries:    {len(entries)}")
    if len(entries) == 0:
        print(f"  No entries for speaker '{args.speaker}'")
        sys.exit(1)

    if not os.path.isdir(WAVS_DIR):
        print(f"\nERROR: wavs/ not found at {WAVS_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = []
    skipped_missing = 0
    for e in entries:
        wav_path = os.path.join(WAVS_DIR, f"{e['wav_id']}.wav")
        output_path = os.path.join(OUTPUT_DIR, f"{e['wav_id']}.wav")
        if not os.path.isfile(wav_path):
            skipped_missing += 1
            continue
        tasks.append((e["wav_id"], wav_path, output_path, args.denoise, args.prop))

    if skipped_missing > 0:
        print(f"  WARNING: {skipped_missing} audio files not found")
    print(f"  Processing {len(tasks)} files...")

    results = {"ok": [], "rejected": [], "error": []}
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, t): t[0] for t in tasks}
        for future in as_completed(futures):
            wav_id, status, reason = future.result()
            results[status].append((wav_id, reason))
            completed += 1
            if completed % 100 == 0:
                print(f"  Processed {completed}/{len(tasks)}...")

    ok_ids = {wav_id for wav_id, _ in results["ok"]}
    clean_metadata_path = os.path.join(DATASET_ROOT, "metadata_clean.csv")
    with open(clean_metadata_path, "w", encoding="utf-8") as f:
        for e in entries:
            if e["wav_id"] in ok_ids:
                f.write(f"{e['wav_id']}|{e['roman']}|{e['sinhala']}|{e['speaker']}\n")

    report_path = os.path.join(DATASET_ROOT, "preprocessing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Audio Preprocessing Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Speaker: {args.speaker} | Denoise: {args.denoise} (prop={args.prop})\n\n")
        f.write(f"Accepted: {len(results['ok'])}\n")
        f.write(f"Rejected: {len(results['rejected'])}\n")
        f.write(f"Errors:   {len(results['error'])}\n\n")
        if results["rejected"]:
            f.write(f"REJECTED\n{'-'*40}\n")
            for wav_id, reason in sorted(results["rejected"]):
                f.write(f"  {wav_id}: {reason}\n")
        if results["error"]:
            f.write(f"\nERRORS\n{'-'*40}\n")
            for wav_id, reason in sorted(results["error"]):
                f.write(f"  {wav_id}: {reason}\n")
        low_snr = [(wid, r) for wid, r in results["ok"] if "LOW_SNR" in r]
        if low_snr:
            f.write(f"\nLOW SNR WARNINGS\n{'-'*40}\n")
            for wav_id, reason in sorted(low_snr):
                f.write(f"  {wav_id}: {reason}\n")

    reject_pct = len(results["rejected"]) / max(len(tasks), 1) * 100
    print(f"\n  Accepted: {len(results['ok'])}")
    print(f"  Rejected: {len(results['rejected'])} ({reject_pct:.1f}%)")
    print(f"  Errors:   {len(results['error'])}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Metadata: {clean_metadata_path}")
    print(f"  Report:   {report_path}")
    if reject_pct > 10:
        print(f"\n  WARNING: Reject rate {reject_pct:.1f}% > 10%. Consider relaxing thresholds.")

if __name__ == "__main__":
    main()
PYTHON_PREPROCESS_EOF

# ===== prepare_data.py (IMPROVED) =====
cat > "$TRAIN_DIR/prepare_data.py" << 'PYTHON_PREPARE_EOF'
"""
Sinhala TTS Dataset Preparation Script
Includes text cleaning and quality filtering.
"""
import os, re, random, shutil
from pathlib import Path

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")
METADATA_CLEAN_FILE = os.path.join(DATASET_ROOT, "metadata_clean.csv")
WAVS_DIR = os.path.join(DATASET_ROOT, "wavs")
WAVS_PROCESSED_DIR = os.path.join(DATASET_ROOT, "wavs_processed")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "training", "dataset")
MULTISPEAKER_DIR = os.path.join(OUTPUT_DIR, "multispeaker")
FEMALE_ONLY_DIR = os.path.join(OUTPUT_DIR, "female_only")
FEMALE_ONLY_CLEAN_DIR = os.path.join(OUTPUT_DIR, "female_only_clean")

VAL_SPLIT = 0.05
VAL_SPLIT_CLEAN = 0.10


def clean_text(text):
    text = re.sub(r'[\"\"\"\'\'\'\\u2018\\u2019\\u201C\\u201D]', "'", text)
    text = re.sub(r'[\\u2013\\u2014\\u2015]+', "-", text)
    text = re.sub(r'\\s+', " ", text)
    return text.strip()


def parse_metadata(metadata_path, filter_quality=False):
    entries = []
    seen_texts = set()
    skipped_short = 0
    skipped_dup = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) < 4: continue
            wav_id, roman, sinhala, speaker = parts[0], parts[1], parts[2], parts[3]
            roman = clean_text(roman)

            if filter_quality:
                if len(roman) < 10:
                    skipped_short += 1
                    continue
                if roman in seen_texts:
                    skipped_dup += 1
                    continue
                seen_texts.add(roman)

            entries.append({"wav_id": wav_id, "roman": roman, "sinhala": sinhala, "speaker": speaker})

    if filter_quality and (skipped_short > 0 or skipped_dup > 0):
        print(f"  Quality filter: skipped {skipped_short} short, {skipped_dup} duplicates")
    return entries


def verify_audio(entries, wavs_dir):
    valid, missing = [], 0
    for e in entries:
        if os.path.isfile(os.path.join(wavs_dir, f"{e['wav_id']}.wav")):
            valid.append(e)
        else:
            missing += 1
    if missing: print(f"  WARNING: {missing} audio files not found")
    return valid


def create_split(entries, output_dir, wavs_dir, use_sinhala=False, val_split=None):
    if val_split is None:
        val_split = VAL_SPLIT
    os.makedirs(output_dir, exist_ok=True)

    wavs_link = os.path.join(output_dir, "wavs")
    if os.path.exists(wavs_link):
        if os.path.islink(wavs_link): os.unlink(wavs_link)
        elif os.path.isdir(wavs_link): shutil.rmtree(wavs_link)
    os.symlink(wavs_dir, wavs_link)

    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split))
    val_entries, train_entries = shuffled[:val_count], shuffled[val_count:]
    text_key = "sinhala" if use_sinhala else "roman"
    multi = len(set(x["speaker"] for x in entries)) > 1

    for name, data in [("train", train_entries), ("val", val_entries)]:
        path = os.path.join(output_dir, f"metadata_{name}.csv")
        with open(path, "w", encoding="utf-8") as f:
            for e in data:
                t = e[text_key]
                if multi:
                    f.write(f"{e['wav_id']}|{t}|{t}|{e['speaker']}\n")
                else:
                    f.write(f"{e['wav_id']}|{t}|{t}\n")
        print(f"  {name}: {len(data)} entries -> {path}")


print("=" * 60)
print("  Sinhala TTS Dataset Preparation")
print("=" * 60)
all_entries = parse_metadata(METADATA_FILE)
print(f"Total entries: {len(all_entries)}")
speakers = {}
for e in all_entries:
    speakers[e["speaker"]] = speakers.get(e["speaker"], 0) + 1
for s, c in speakers.items():
    print(f"  Speaker '{s}': {c}")

valid = verify_audio(all_entries, WAVS_DIR)
print(f"Valid with audio: {len(valid)}")
if not valid:
    print("ERROR: No audio files!"); exit(1)

print("\n--- Multi-Speaker Dataset ---")
create_split(valid, MULTISPEAKER_DIR, WAVS_DIR)

female = [e for e in valid if e["speaker"] == "oshadi"]
print(f"\n--- Female-Only Dataset ({len(female)} clips) ---")
create_split(female, FEMALE_ONLY_DIR, WAVS_DIR)

# CLEAN dataset variant (preprocessed audio)
if os.path.exists(METADATA_CLEAN_FILE) and os.path.isdir(WAVS_PROCESSED_DIR):
    print(f"\n--- Female-Only CLEAN Dataset (preprocessed audio) ---")
    clean_entries = parse_metadata(METADATA_CLEAN_FILE, filter_quality=True)
    female_clean = [e for e in clean_entries if e["speaker"] == "oshadi"]
    print(f"  Clean female entries: {len(female_clean)}")
    female_clean = verify_audio(female_clean, WAVS_PROCESSED_DIR)
    create_split(female_clean, FEMALE_ONLY_CLEAN_DIR, WAVS_PROCESSED_DIR, val_split=VAL_SPLIT_CLEAN)
else:
    print("\n--- Skipping CLEAN dataset (run preprocess_audio.py first) ---")

print("\nDone!")
PYTHON_PREPARE_EOF

# ===== train_female_only.py (IMPROVED) =====
cat > "$TRAIN_DIR/train_female_only.py" << 'PYTHON_TRAIN_EOF'
"""
Sinhala TTS - Female-Only VITS Training (Improved)
===================================================
Single-speaker VITS on preprocessed oshadi data.
Key improvements: mel_fmax=8000, add_blank, lower LR, FP32.
"""
import os, sys

DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(DATASET_ROOT, "training")
DATASET_PATH = os.path.join(TRAINING_DIR, "dataset", "female_only_clean")
OUTPUT_PATH = os.path.join(TRAINING_DIR, "output_female_only_clean")

BATCH_SIZE = 16          # Smaller batch = more gradient updates per epoch with small data
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3000        # More epochs needed for small dataset
LEARNING_RATE = 0.0001   # Lower LR prevents overshooting with limited data
SAMPLE_RATE = 22050
NUM_WORKERS = 4

CHARACTERS = " !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ"

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainerArgs


def main():
    print("=" * 60)
    print("  Sinhala VITS Female-Only Training (Improved)")
    print("=" * 60)

    train_meta = os.path.join(DATASET_PATH, "metadata_train.csv")
    if not os.path.exists(train_meta):
        print(f"\nERROR: {train_meta} not found. Run prepare_data.py first!")
        sys.exit(1)

    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=8000,  # Limit to speech-relevant frequencies (was None/11025Hz)
    )

    character_config = CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        characters=CHARACTERS,
        punctuations="!'(),-.:;=?",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
    )

    config = VitsConfig(
        output_path=OUTPUT_PATH,
        run_name="sinhala-vits-female-clean",

        audio=audio_config,

        # Single speaker
        use_speaker_embedding=False,
        num_speakers=0,

        characters=character_config,
        text_cleaner=None,
        use_phonemes=False,
        add_blank=True,  # Insert blank tokens between chars for better alignment

        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=NUM_WORKERS,
        num_eval_loader_workers=2,
        epochs=NUM_EPOCHS,

        lr_gen=LEARNING_RATE,
        lr_disc=LEARNING_RATE,
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.99995, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.99995, "last_epoch": -1},

        print_step=25,
        plot_step=100,
        print_eval=True,
        mixed_precision=False,  # FP32 for stability with small dataset
        save_step=1000,
        save_n_checkpoints=5,
        save_best_after=3000,

        run_eval=True,
        test_delay_epochs=3,

        datasets=[{
            "formatter": "ljspeech",
            "meta_file_train": "metadata_train.csv",
            "meta_file_val": "metadata_val.csv",
            "path": DATASET_PATH,
            "language": "si",
        }],

        test_sentences=[
            ["ayubovan, mama ōśadī."],
            ["mē siṁhala parigaṇaka handa ha'ḍunaganīma sanḍahā vū parikşaṇayak."],
            ["āyuşmat bakkula teraṇuvō"],
            ["śrī laṁkāvē rājya bhāşāva siṁhala ya."],
        ],

        cudnn_benchmark=True,
    )

    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_batch_size * 10,
        eval_split_size=0.05,
    )

    print(f"\nLoaded {len(train_samples)} training / {len(eval_samples)} eval samples")

    model = Vits.init_from_config(config)

    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, gpu=0),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting training...")
    print(f"Monitor: tensorboard --logdir {OUTPUT_PATH} --bind_all")
    trainer.fit()


if __name__ == "__main__":
    main()
PYTHON_TRAIN_EOF

# ===== inference.py (IMPROVED with post-processing) =====
cat > "$TRAIN_DIR/inference.py" << 'PYTHON_INFERENCE_EOF'
"""
Sinhala TTS - Inference with Post-Processing
"""
import os, sys, argparse, torch, numpy as np, soundfile as sf

try:
    import pyloudnorm as pyln
    from scipy.signal import butter, sosfilt
    HAS_POSTPROCESS = True
except ImportError:
    HAS_POSTPROCESS = False


def post_process(wav_array, sr):
    """Highpass filter + loudness normalization."""
    if not HAS_POSTPROCESS:
        return wav_array
    wav = np.array(wav_array, dtype=np.float32)
    # Highpass at 80Hz
    sos = butter(5, 80, btype="high", fs=sr, output="sos")
    wav = sosfilt(sos, wav).astype(np.float32)
    # Loudness normalize to -20 LUFS
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    if loudness > -70:
        wav = pyln.normalize.loudness(wav, loudness, -20.0)
    wav = np.clip(wav, -1.0, 1.0)
    return wav


def find_best_model(output_dir):
    for root, dirs, files in os.walk(output_dir):
        if "best_model.pth" in files:
            m = os.path.join(root, "best_model.pth")
            c = os.path.join(root, "config.json")
            if os.path.exists(c): return m, c
    cks = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith("checkpoint_") and f.endswith(".pth"):
                cks.append(os.path.join(root, f))
    if cks:
        cks.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        m = cks[0]; c = os.path.join(os.path.dirname(m), "config.json")
        if os.path.exists(c): return m, c
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Sinhala TTS Inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--speaker", type=str, default="oshadi")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()

    model_path, config_path = args.model, args.config
    if model_path is None:
        td = os.path.dirname(os.path.abspath(__file__))
        for d in [args.model_dir,
                  os.path.join(td, "output_female_only_clean"),
                  os.path.join(td, "output"),
                  os.path.join(td, "output_female_only")]:
            if d and os.path.isdir(d):
                model_path, config_path = find_best_model(d)
                if model_path: break
    if not model_path or not os.path.exists(model_path):
        print("ERROR: No model found."); sys.exit(1)

    print(f"Model: {model_path}")
    from TTS.utils.synthesizer import Synthesizer
    synth = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path,
                        use_cuda=torch.cuda.is_available())
    wav = synth.tts(text=args.text, speaker_name=args.speaker)
    sr = synth.tts_config.audio.sample_rate
    wav_array = np.array(wav)
    if not args.no_postprocess:
        wav_array = post_process(wav_array, sr)
    sf.write(args.output, wav_array, sr)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
PYTHON_INFERENCE_EOF

echo "  Training scripts created."

# ---- STEP 6: Preprocess audio ----
echo ""
echo "[6/8] Preprocessing oshadi audio (loudness normalization + trimming)..."
cd "$PROJECT_DIR"
python training/preprocess_audio.py --speaker oshadi
echo ""
echo "  Check preprocessing_report.txt for details."

# ---- STEP 7: Prepare cleaned dataset ----
echo ""
echo "[7/8] Preparing cleaned dataset..."
python training/prepare_data.py

# ---- STEP 8: Summary ----
echo ""
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "  To start training:"
echo ""
echo "    tmux new -s tts"
echo "    source /workspace/sinhala-tts/tts_env/bin/activate"
echo "    cd /workspace/sinhala-tts/sinhala-tts-dataset"
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
