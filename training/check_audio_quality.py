"""
Check audio file quality on the server.
Run this ON THE SERVER where wav files exist:
  python check_audio_quality.py
"""
import os
import wave
import glob
import struct
import math

WAVS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wavs")
METADATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata.csv")


def get_wav_info(path):
    """Get duration, sample rate, RMS energy from a WAV file."""
    try:
        with wave.open(path, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration = frames / rate

            # Read raw audio for RMS calculation (first 5s max)
            max_frames = min(frames, rate * 5)
            raw = wf.readframes(max_frames)

            if sampwidth == 2:
                fmt = f"<{max_frames * channels}h"
                try:
                    samples = struct.unpack(fmt, raw)
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0
                except:
                    rms = -1
            else:
                rms = -1

            return {
                "duration": duration,
                "rate": rate,
                "channels": channels,
                "sampwidth": sampwidth,
                "rms": rms,
            }
    except Exception as e:
        return {"error": str(e)}


def main():
    # Get expected wav IDs from metadata
    expected_ids = set()
    with open(METADATA, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                expected_ids.add(parts[0])

    print("=" * 60)
    print("AUDIO FILE QUALITY CHECK")
    print("=" * 60)
    print(f"Expected files from metadata: {len(expected_ids)}")

    # Check which files exist
    existing_files = set()
    for wav_path in glob.glob(os.path.join(WAVS_DIR, "*.wav")):
        wav_id = os.path.splitext(os.path.basename(wav_path))[0]
        existing_files.add(wav_id)

    missing = expected_ids - existing_files
    extra = existing_files - expected_ids

    print(f"Existing WAV files: {len(existing_files)}")
    print(f"Missing files: {len(missing)}")
    if missing:
        for m in sorted(missing)[:10]:
            print(f"  MISSING: {m}.wav")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"Extra files (not in metadata): {len(extra)}")

    # Analyze audio properties
    print(f"\nAnalyzing audio files...")
    durations = []
    sample_rates = set()
    channels_set = set()
    too_short = []  # < 1s
    too_long = []   # > 15s
    very_quiet = []  # low RMS
    errors = []

    wav_files = sorted(glob.glob(os.path.join(WAVS_DIR, "*.wav")))
    total = len(wav_files)

    for i, wav_path in enumerate(wav_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{total}...")

        info = get_wav_info(wav_path)
        wav_id = os.path.splitext(os.path.basename(wav_path))[0]

        if "error" in info:
            errors.append((wav_id, info["error"]))
            continue

        durations.append(info["duration"])
        sample_rates.add(info["rate"])
        channels_set.add(info["channels"])

        if info["duration"] < 1.0:
            too_short.append((wav_id, info["duration"]))
        if info["duration"] > 15.0:
            too_long.append((wav_id, info["duration"]))
        if 0 <= info["rms"] < 100:
            very_quiet.append((wav_id, info["rms"]))

    # Report
    print(f"\n{'='*60}")
    print("AUDIO ANALYSIS RESULTS")
    print(f"{'='*60}")

    if durations:
        durations.sort()
        total_dur = sum(durations)
        print(f"\nTotal audio duration: {total_dur/3600:.2f} hours")
        print(f"Duration stats:")
        print(f"  Min:    {min(durations):.2f}s")
        print(f"  Max:    {max(durations):.2f}s")
        print(f"  Mean:   {total_dur/len(durations):.2f}s")
        print(f"  Median: {durations[len(durations)//2]:.2f}s")
        print(f"  P5:     {durations[int(len(durations)*0.05)]:.2f}s")
        print(f"  P95:    {durations[int(len(durations)*0.95)]:.2f}s")

    print(f"\nSample rates found: {sample_rates}")
    print(f"Channel configs found: {channels_set}")

    print(f"\n{'='*60}")
    print("ISSUES FOUND")
    print(f"{'='*60}")

    print(f"\n1. Clips too short (<1s): {len(too_short)}")
    if too_short:
        for wid, dur in too_short[:10]:
            print(f"   {wid}: {dur:.2f}s")
        if len(too_short) > 10:
            print(f"   ... and {len(too_short) - 10} more")

    print(f"\n2. Clips too long (>15s): {len(too_long)}")
    if too_long:
        for wid, dur in too_long[:10]:
            print(f"   {wid}: {dur:.2f}s")
        if len(too_long) > 10:
            print(f"   ... and {len(too_long) - 10} more")

    print(f"\n3. Very quiet clips (low RMS): {len(very_quiet)}")
    if very_quiet:
        for wid, rms in very_quiet[:10]:
            print(f"   {wid}: RMS={rms:.1f}")

    print(f"\n4. Broken/unreadable files: {len(errors)}")
    if errors:
        for wid, err in errors[:10]:
            print(f"   {wid}: {err}")

    # Final score
    print(f"\n{'='*60}")
    print("QUALITY SCORE")
    print(f"{'='*60}")

    total_issues = len(too_short) + len(too_long) + len(very_quiet) + len(errors) + len(missing)
    pct_clean = (1 - total_issues / max(len(existing_files), 1)) * 100

    print(f"  Clean samples: {pct_clean:.1f}%")
    print(f"  Total issues: {total_issues}")

    if total_dur / 3600 < 2:
        print(f"\n  ⚠️  CRITICAL: Only {total_dur/3600:.1f} hours of data. Need 5+ hours minimum!")
    elif total_dur / 3600 < 5:
        print(f"\n  ⚠️  WARNING: {total_dur/3600:.1f} hours is low. 10+ hours recommended for VITS.")
    else:
        print(f"\n  ✓ Data quantity OK: {total_dur/3600:.1f} hours")


if __name__ == "__main__":
    main()
