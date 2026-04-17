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
import csv
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

# ── Configuration ────────────────────────────────────────────────────
DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.csv")
WAVS_DIR = os.path.join(DATASET_ROOT, "wavs")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "wavs_processed")
SAMPLE_RATE = 22050

# Quality thresholds
TARGET_LUFS = -23.0         # EBU R128 standard loudness
MIN_DURATION = 1.5          # seconds
MAX_DURATION = 12.0         # seconds
MIN_RMS = 0.01              # float32 RMS threshold
MIN_SNR_DB = 15.0           # estimated SNR threshold
TRIM_TOP_DB = 25            # librosa silence trim threshold


def estimate_snr(audio, sr):
    """Estimate SNR by comparing voiced frames vs noise floor."""
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    # Compute frame-level RMS energy
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.mean(frames ** 2, axis=0)

    if len(frame_energy) == 0:
        return 0.0

    # Use 20th percentile as noise floor estimate
    noise_floor = np.percentile(frame_energy, 20)
    signal_energy = np.mean(frame_energy)

    if noise_floor <= 0:
        return 60.0  # effectively clean

    snr = 10 * np.log10(signal_energy / noise_floor)
    return float(snr)


def process_single_file(args_tuple):
    """Process a single WAV file. Returns (wav_id, status, reason) tuple."""
    wav_id, wav_path, output_path, do_denoise, denoise_prop = args_tuple

    try:
        # Load audio
        audio, sr = sf.read(wav_path, dtype="float32")

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # 1. Silence trimming
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)

        # 2. Duration check (after trimming)
        duration = len(audio_trimmed) / sr
        if duration < MIN_DURATION:
            return (wav_id, "rejected", f"too_short: {duration:.2f}s < {MIN_DURATION}s")
        if duration > MAX_DURATION:
            return (wav_id, "rejected", f"too_long: {duration:.2f}s > {MAX_DURATION}s")

        # 3. RMS check
        rms = np.sqrt(np.mean(audio_trimmed ** 2))
        if rms < MIN_RMS:
            return (wav_id, "rejected", f"too_quiet: RMS={rms:.4f} < {MIN_RMS}")

        # 4. SNR estimation
        snr = estimate_snr(audio_trimmed, sr)
        snr_warning = ""
        if snr < MIN_SNR_DB:
            snr_warning = f" [LOW_SNR: {snr:.1f}dB]"

        # 5. Optional denoising
        if do_denoise:
            import noisereduce as nr
            audio_trimmed = nr.reduce_noise(
                y=audio_trimmed, sr=sr,
                prop_decrease=denoise_prop,
                stationary=True,
            )

        # 6. Loudness normalization to target LUFS
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_trimmed)
        if loudness < -70.0:
            return (wav_id, "rejected", f"silence: loudness={loudness:.1f} LUFS")
        audio_normalized = pyln.normalize.loudness(audio_trimmed, loudness, TARGET_LUFS)

        # Clip to prevent clipping after normalization
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        # 7. Save processed file
        sf.write(output_path, audio_normalized, sr, subtype="PCM_16")

        final_duration = len(audio_normalized) / sr
        return (wav_id, "ok", f"dur={final_duration:.2f}s rms={rms:.3f} snr={snr:.1f}dB{snr_warning}")

    except Exception as e:
        return (wav_id, "error", str(e))


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for TTS training")
    parser.add_argument("--speaker", type=str, default="oshadi",
                        help="Speaker to process: 'oshadi', 'mettananda', or 'all' (default: oshadi)")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable spectral denoising (conservative)")
    parser.add_argument("--prop", type=float, default=0.5,
                        help="Denoising strength 0.0-1.0 (default: 0.5, higher = more aggressive)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Sinhala TTS Audio Preprocessing")
    print("=" * 60)
    print(f"\n  Speaker filter:  {args.speaker}")
    print(f"  Denoise:         {args.denoise} (prop={args.prop})")
    print(f"  Target LUFS:     {TARGET_LUFS}")
    print(f"  Duration range:  {MIN_DURATION}s - {MAX_DURATION}s")
    print(f"  Min SNR:         {MIN_SNR_DB}dB (warning only)")
    print(f"  Trim threshold:  {TRIM_TOP_DB}dB")
    print(f"  Workers:         {args.workers}")

    # Parse metadata to get file list
    if not os.path.exists(METADATA_FILE):
        print(f"\nERROR: metadata.csv not found at {METADATA_FILE}")
        sys.exit(1)

    entries = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            wav_id, roman, sinhala, speaker = parts[0], parts[1], parts[2], parts[3]
            if args.speaker != "all" and speaker != args.speaker:
                continue
            entries.append({
                "wav_id": wav_id,
                "roman": roman,
                "sinhala": sinhala,
                "speaker": speaker,
            })

    print(f"\n  Entries to process: {len(entries)}")

    if len(entries) == 0:
        print(f"  No entries found for speaker '{args.speaker}'")
        sys.exit(1)

    # Check source wavs exist
    if not os.path.isdir(WAVS_DIR):
        print(f"\nERROR: wavs/ directory not found at {WAVS_DIR}")
        print("Download and extract wavs.tar.gz first.")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build task list
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
        print(f"  WARNING: {skipped_missing} audio files not found (skipped)")

    print(f"  Files to process: {len(tasks)}")
    print(f"\nProcessing...")

    # Process in parallel
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

    # Generate cleaned metadata
    ok_ids = {wav_id for wav_id, _ in results["ok"]}
    clean_metadata_path = os.path.join(DATASET_ROOT, "metadata_clean.csv")

    with open(clean_metadata_path, "w", encoding="utf-8") as f:
        for e in entries:
            if e["wav_id"] in ok_ids:
                f.write(f"{e['wav_id']}|{e['roman']}|{e['sinhala']}|{e['speaker']}\n")

    # Generate report
    report_path = os.path.join(DATASET_ROOT, "preprocessing_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Sinhala TTS Audio Preprocessing Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Speaker filter: {args.speaker}\n")
        f.write(f"Denoise: {args.denoise} (prop={args.prop})\n")
        f.write(f"Target LUFS: {TARGET_LUFS}\n\n")

        f.write(f"SUMMARY\n{'-' * 40}\n")
        f.write(f"Total processed: {len(tasks)}\n")
        f.write(f"Accepted:        {len(results['ok'])}\n")
        f.write(f"Rejected:        {len(results['rejected'])}\n")
        f.write(f"Errors:          {len(results['error'])}\n")
        reject_pct = len(results["rejected"]) / max(len(tasks), 1) * 100
        f.write(f"Reject rate:     {reject_pct:.1f}%\n\n")

        if results["rejected"]:
            f.write(f"REJECTED FILES\n{'-' * 40}\n")
            for wav_id, reason in sorted(results["rejected"]):
                f.write(f"  {wav_id}: {reason}\n")
            f.write("\n")

        if results["error"]:
            f.write(f"ERRORS\n{'-' * 40}\n")
            for wav_id, reason in sorted(results["error"]):
                f.write(f"  {wav_id}: {reason}\n")
            f.write("\n")

        # Log low-SNR warnings from accepted files
        low_snr = [(wid, r) for wid, r in results["ok"] if "LOW_SNR" in r]
        if low_snr:
            f.write(f"LOW SNR WARNINGS (accepted but noisy)\n{'-' * 40}\n")
            for wav_id, reason in sorted(low_snr):
                f.write(f"  {wav_id}: {reason}\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Accepted:  {len(results['ok'])}")
    print(f"  Rejected:  {len(results['rejected'])} ({reject_pct:.1f}%)")
    print(f"  Errors:    {len(results['error'])}")
    print(f"\n  Output WAVs:     {OUTPUT_DIR}")
    print(f"  Clean metadata:  {clean_metadata_path}")
    print(f"  Full report:     {report_path}")

    if reject_pct > 10:
        print(f"\n  WARNING: Reject rate is {reject_pct:.1f}% (>10%).")
        print(f"  Consider relaxing thresholds if too many clips lost.")

    low_snr_count = len([1 for _, r in results["ok"] if "LOW_SNR" in r])
    if low_snr_count > 0:
        print(f"\n  NOTE: {low_snr_count} accepted clips have low SNR (<{MIN_SNR_DB}dB).")
        print(f"  Consider re-running with --denoise to reduce noise.")


if __name__ == "__main__":
    main()
