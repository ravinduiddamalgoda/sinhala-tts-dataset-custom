#!/bin/bash
# ================================================================================
# SINHALA TTS - COMPLETE SERVER SETUP & TRAINING (Copy-paste this into vast.ai)
# ================================================================================
# This is a SELF-CONTAINED script. It:
#   1. Installs all dependencies
#   2. Clones the original dataset repo (code only, ~small)
#   3. Downloads audio files from GitHub Releases (~1.5GB)
#   4. Creates training scripts inline (no need to push your own repo)
#   5. Prepares data and starts training
#
# Usage: Just paste this entire script into your vast.ai terminal
# ================================================================================
set -e

echo "=========================================="
echo "  Sinhala TTS - Complete Setup"
echo "=========================================="

# ---- STEP 1: System dependencies ----
echo "[1/6] Installing system dependencies..."
apt-get update && apt-get install -y \
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
    echo "[2/6] Cloning dataset repository..."
    git clone https://github.com/pnfo/sinhala-tts-dataset.git
else
    echo "[2/6] Dataset repo already exists."
fi

PROJECT_DIR="$WORK_DIR/sinhala-tts-dataset"
cd "$PROJECT_DIR"

# ---- STEP 3: Download audio files from GitHub Releases ----
if [ ! -d "wavs" ] || [ "$(ls wavs/*.wav 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo "[3/6] Downloading audio files from GitHub Release (~1.5GB)..."
    wget --show-progress "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz" -O wavs.tar.gz
    tar -xzf wavs.tar.gz
    rm -f wavs.tar.gz
else
    echo "[3/6] Audio files already present."
fi
echo "  Found $(ls wavs/*.wav 2>/dev/null | wc -l) wav files"

# ---- STEP 4: Python environment ----
echo "[4/6] Setting up Python environment..."
cd "$WORK_DIR"
if [ ! -d "tts_env" ]; then
    python3 -m venv tts_env
fi
source tts_env/bin/activate
pip install --upgrade pip setuptools wheel -q

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
pip install TTS==0.22.0 -q
pip install matplotlib tensorboard pandas numpy scipy librosa soundfile Unidecode inflect tqdm gradio -q

# ---- STEP 5: Create training scripts ----
echo "[5/6] Creating training scripts..."
TRAIN_DIR="$PROJECT_DIR/training"
mkdir -p "$TRAIN_DIR"

# ===== prepare_data.py =====
cat > "$TRAIN_DIR/prepare_data.py" << 'PYTHON_EOF'
import os, csv, random, shutil
from pathlib import Path

DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")
WAVS_DIR = os.path.join(DATASET_ROOT, "wavs")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "training", "dataset")
MULTISPEAKER_DIR = os.path.join(OUTPUT_DIR, "multispeaker")
FEMALE_ONLY_DIR = os.path.join(OUTPUT_DIR, "female_only")
VAL_SPLIT = 0.05

def parse_metadata(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            if len(parts) < 4: continue
            entries.append({"wav_id": parts[0], "roman": parts[1], "sinhala": parts[2], "speaker": parts[3]})
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

def create_split(entries, output_dir, wavs_dir, use_sinhala=False):
    os.makedirs(output_dir, exist_ok=True)
    wavs_link = os.path.join(output_dir, "wavs")
    if os.path.exists(wavs_link):
        if os.path.islink(wavs_link): os.unlink(wavs_link)
        elif os.path.isdir(wavs_link): shutil.rmtree(wavs_link)
    os.symlink(wavs_dir, wavs_link)

    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * VAL_SPLIT))
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
    print("ERROR: No audio files! Download wavs from GitHub releases first.")
    exit(1)

print("\n--- Multi-Speaker Dataset ---")
create_split(valid, MULTISPEAKER_DIR, WAVS_DIR)
female = [e for e in valid if e["speaker"] == "oshadi"]
print(f"\n--- Female-Only Dataset ({len(female)} clips) ---")
create_split(female, FEMALE_ONLY_DIR, WAVS_DIR)
print("\nDone!")
PYTHON_EOF

# ===== train.py =====
cat > "$TRAIN_DIR/train.py" << 'PYTHON_EOF'
import os, sys

DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(DATASET_ROOT, "training")
DATASET_PATH = os.path.join(TRAINING_DIR, "dataset", "multispeaker")
OUTPUT_PATH = os.path.join(TRAINING_DIR, "output")

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 0.0002
SAMPLE_RATE = 22050
CHARACTERS = " !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ"

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainerArgs

def main():
    print("=" * 60)
    print("  Sinhala VITS Multi-Speaker Training")
    print("=" * 60)

    train_meta = os.path.join(DATASET_PATH, "metadata_train.csv")
    if not os.path.exists(train_meta):
        print(f"ERROR: {train_meta} not found. Run prepare_data.py first!")
        sys.exit(1)

    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE, win_length=1024, hop_length=256,
        num_mels=80, mel_fmin=0, mel_fmax=None,
    )
    character_config = CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        characters=CHARACTERS, punctuations="!'(),-.:;=?",
        pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
    )
    config = VitsConfig(
        output_path=OUTPUT_PATH,
        run_name="sinhala-vits-multispeaker",
        audio=audio_config,
        use_speaker_embedding=True, num_speakers=2,
        characters=character_config, text_cleaner=None, use_phonemes=False,
        batch_size=BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=8, num_eval_loader_workers=4,
        epochs=NUM_EPOCHS,
        lr_gen=LEARNING_RATE, lr_disc=LEARNING_RATE,
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},
        print_step=50, plot_step=100, print_eval=True,
        mixed_precision=True, save_step=5000,
        save_n_checkpoints=3, save_best_after=10000,
        run_eval=True, test_delay_epochs=5,
        datasets=[{
            "formatter": "ljspeech",
            "meta_file_train": "metadata_train.csv",
            "meta_file_val": "metadata_val.csv",
            "path": DATASET_PATH,
            "language": "si",
        }],
        test_sentences=[
            ["ayubovan, mama ōśadī.", "oshadi", None, None],
            ["mē siṁhala parigaṇaka handa.", "oshadi", None, None],
            ["śrī laṁkāvē rājya bhāşāva siṁhala ya.", "oshadi", None, None],
            ["apē raṭē kāntāvanṭa dinā gatu yutu ayitivāsikam ræsak tibenavā.", "oshadi", None, None],
        ],
        cudnn_benchmark=True,
    )

    train_samples, eval_samples = load_tts_samples(
        config.datasets[0], eval_split=True,
        eval_split_max_size=config.eval_batch_size * 10, eval_split_size=0.05,
    )
    print(f"Loaded {len(train_samples)} train / {len(eval_samples)} eval samples")

    model = Vits.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, gpu=0),
        config, output_path=OUTPUT_PATH, model=model,
        train_samples=train_samples, eval_samples=eval_samples,
    )
    print("\nStarting training...")
    print(f"Monitor: tensorboard --logdir {OUTPUT_PATH} --bind_all")
    trainer.fit()

if __name__ == "__main__":
    main()
PYTHON_EOF

# ===== inference.py =====
cat > "$TRAIN_DIR/inference.py" << 'PYTHON_EOF'
import os, sys, argparse, torch, numpy as np, soundfile as sf

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
    args = parser.parse_args()

    model_path, config_path = args.model, args.config
    if model_path is None:
        td = os.path.dirname(os.path.abspath(__file__))
        for d in [args.model_dir, os.path.join(td, "output"), os.path.join(td, "output_female_only")]:
            if d and os.path.isdir(d):
                model_path, config_path = find_best_model(d)
                if model_path: break
    if not model_path or not os.path.exists(model_path):
        print("ERROR: No model found. Specify --model or --model-dir"); sys.exit(1)

    from TTS.utils.synthesizer import Synthesizer
    synth = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path, use_cuda=torch.cuda.is_available())
    wav = synth.tts(text=args.text, speaker_name=args.speaker)
    sf.write(args.output, np.array(wav), synth.tts_config.audio.sample_rate)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
PYTHON_EOF

# ===== demo.py =====
cat > "$TRAIN_DIR/demo.py" << 'PYTHON_EOF'
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import find_best_model

def create_demo():
    import gradio as gr
    from TTS.utils.synthesizer import Synthesizer

    td = os.path.dirname(os.path.abspath(__file__))
    model_path, config_path = None, None
    for d in ["output", "output_female_only"]:
        sd = os.path.join(td, d)
        if os.path.isdir(sd):
            model_path, config_path = find_best_model(sd)
            if model_path: break
    if not model_path:
        print("ERROR: No trained model found!"); sys.exit(1)

    print(f"Loading: {model_path}")
    synth = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path, use_cuda=torch.cuda.is_available())
    multi = hasattr(synth.tts_model, 'num_speakers') and synth.tts_model.num_speakers > 1

    def synthesize(text, speaker="oshadi"):
        if not text.strip(): return None
        wav = synth.tts(text=text, speaker_name=speaker if multi else None)
        return (synth.tts_config.audio.sample_rate, np.array(wav, dtype=np.float32))

    with gr.Blocks(title="Sinhala TTS") as demo:
        gr.Markdown("# Sinhala Text-to-Speech Demo")
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Romanized Sinhala", lines=3)
                if multi:
                    spk = gr.Dropdown(["oshadi", "mettananda"], value="oshadi", label="Speaker")
                btn = gr.Button("Synthesize", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Output", type="numpy")
        examples = [["ayubōvan, mama ōśadī."], ["apē raṭē kāntāvanṭa dinā gatu yutu ayitivāsikam ræsak tibenavā."]]
        gr.Examples(examples=examples, inputs=[text_in])
        if multi:
            btn.click(fn=synthesize, inputs=[text_in, spk], outputs=[audio_out])
        else:
            btn.click(fn=lambda t: synthesize(t), inputs=[text_in], outputs=[audio_out])
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
PYTHON_EOF

echo "  Training scripts created."

# ---- STEP 6: Prepare data ----
echo "[6/6] Preparing dataset..."
cd "$PROJECT_DIR"
source "$WORK_DIR/tts_env/bin/activate"
python training/prepare_data.py

echo ""
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "  To start training, run:"
echo ""
echo "    tmux new -s tts"
echo "    source /workspace/sinhala-tts/tts_env/bin/activate"
echo "    cd /workspace/sinhala-tts/sinhala-tts-dataset"
echo ""
echo "    # Single GPU:"
echo "    python training/train.py"
echo ""
echo "    # Multi GPU (2x - RECOMMENDED):"
echo "    CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script training/train.py"
echo ""
echo "    # Monitor (in another tmux pane - Ctrl+B then %):"
echo "    tensorboard --logdir training/output --bind_all --port 6006"
echo ""
echo "    # After training, run demo:"
echo "    python training/demo.py"
echo ""
