"""
Sinhala TTS Dataset Preparation Script
=======================================
Prepares the dataset for VITS multi-speaker training via Coqui TTS.
Creates separate metadata files for multi-speaker and female-only training.
"""

import os
import csv
import random
import shutil
from pathlib import Path

# ---------- Configuration ----------
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")
WAVS_DIR = os.path.join(DATASET_ROOT, "wavs")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "training", "dataset")
MULTISPEAKER_DIR = os.path.join(OUTPUT_DIR, "multispeaker")
FEMALE_ONLY_DIR = os.path.join(OUTPUT_DIR, "female_only")

VAL_SPLIT = 0.05  # 5% validation


def parse_metadata(metadata_path):
    """Parse the pipe-separated metadata.csv"""
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            wav_id = parts[0]        # e.g., sinh_0001
            roman = parts[1]          # romanized text
            sinhala = parts[2]        # sinhala text
            speaker = parts[3]        # mettananda or oshadi
            entries.append({
                "wav_id": wav_id,
                "roman": roman,
                "sinhala": sinhala,
                "speaker": speaker
            })
    return entries


def verify_audio_files(entries, wavs_dir):
    """Check which audio files actually exist"""
    valid = []
    missing = 0
    for e in entries:
        wav_path = os.path.join(wavs_dir, f"{e['wav_id']}.wav")
        if os.path.isfile(wav_path):
            valid.append(e)
        else:
            missing += 1
    if missing > 0:
        print(f"  WARNING: {missing} audio files not found (skipped)")
    return valid


def create_dataset_split(entries, output_dir, wavs_dir, use_sinhala=False):
    """
    Create train/val metadata files in LJSpeech format for Coqui TTS.
    For multi-speaker: wav_id|text|text|speaker_name
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create symlink to wavs directory
    wavs_link = os.path.join(output_dir, "wavs")
    if os.path.exists(wavs_link):
        if os.path.islink(wavs_link):
            os.unlink(wavs_link)
        elif os.path.isdir(wavs_link):
            shutil.rmtree(wavs_link)
    os.symlink(wavs_dir, wavs_link)

    # Shuffle and split
    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * VAL_SPLIT))
    val_entries = shuffled[:val_count]
    train_entries = shuffled[val_count:]

    text_key = "sinhala" if use_sinhala else "roman"

    # Write metadata files - Coqui TTS format: wav_id|text|text|speaker
    for split_name, split_entries in [("train", train_entries), ("val", val_entries)]:
        meta_path = os.path.join(output_dir, f"metadata_{split_name}.csv")
        with open(meta_path, "w", encoding="utf-8") as f:
            for e in split_entries:
                text = e[text_key]
                # Format: audio_file|text|text|speaker_name (for multi-speaker)
                if "speaker" in e and len(set(x["speaker"] for x in entries)) > 1:
                    f.write(f"{e['wav_id']}|{text}|{text}|{e['speaker']}\n")
                else:
                    f.write(f"{e['wav_id']}|{text}|{text}\n")
        print(f"  {split_name}: {len(split_entries)} entries -> {meta_path}")

    return train_entries, val_entries


def main():
    print("=" * 60)
    print("  Sinhala TTS Dataset Preparation")
    print("=" * 60)

    # Parse metadata
    print(f"\nReading metadata from: {METADATA_FILE}")
    all_entries = parse_metadata(METADATA_FILE)
    print(f"  Total entries: {len(all_entries)}")

    # Count by speaker
    speakers = {}
    for e in all_entries:
        speakers[e["speaker"]] = speakers.get(e["speaker"], 0) + 1
    for spk, count in speakers.items():
        print(f"  Speaker '{spk}': {count} entries")

    # Verify audio files exist
    print(f"\nVerifying audio files in: {WAVS_DIR}")
    valid_entries = verify_audio_files(all_entries, WAVS_DIR)
    print(f"  Valid entries with audio: {len(valid_entries)}")

    if len(valid_entries) == 0:
        print("\nERROR: No audio files found!")
        print("Make sure to download wavs from the GitHub release:")
        print("  wget https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/wavs.tar.gz")
        print("  tar -xzf wavs.tar.gz")
        return

    # --- Multi-speaker dataset (recommended - uses all data) ---
    print(f"\n--- Multi-Speaker Dataset (ALL data) ---")
    print(f"Output: {MULTISPEAKER_DIR}")
    create_dataset_split(valid_entries, MULTISPEAKER_DIR, WAVS_DIR, use_sinhala=False)

    # --- Female-only dataset ---
    female_entries = [e for e in valid_entries if e["speaker"] == "oshadi"]
    print(f"\n--- Female-Only Dataset (oshadi) ---")
    print(f"Output: {FEMALE_ONLY_DIR}")
    print(f"  Female entries: {len(female_entries)}")
    create_dataset_split(female_entries, FEMALE_ONLY_DIR, WAVS_DIR, use_sinhala=False)

    # --- Also create Sinhala script versions ---
    print(f"\n--- Multi-Speaker Dataset (Sinhala script) ---")
    sinhala_dir = os.path.join(OUTPUT_DIR, "multispeaker_sinhala")
    create_dataset_split(valid_entries, sinhala_dir, WAVS_DIR, use_sinhala=True)

    print(f"\n--- Female-Only Dataset (Sinhala script) ---")
    sinhala_female_dir = os.path.join(OUTPUT_DIR, "female_only_sinhala")
    create_dataset_split(female_entries, sinhala_female_dir, WAVS_DIR, use_sinhala=True)

    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print("=" * 60)
    print(f"\nRecommended: Use multi-speaker dataset for training:")
    print(f"  {MULTISPEAKER_DIR}")
    print(f"\nThen synthesize with speaker='oshadi' for female voice")


if __name__ == "__main__":
    main()
