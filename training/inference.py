"""
Sinhala TTS - Inference Script
==============================
Load a trained VITS model and synthesize speech.
Supports both multi-speaker and single-speaker models.
"""

import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    import pyloudnorm as pyln
    from scipy.signal import butter, sosfilt
    HAS_POSTPROCESS = True
except ImportError:
    HAS_POSTPROCESS = False


def post_process(wav_array, sr):
    """Apply post-processing to improve synthesized audio quality."""
    if not HAS_POSTPROCESS:
        return wav_array

    wav = np.array(wav_array, dtype=np.float32)

    # Highpass filter at 80Hz to remove low-frequency rumble
    sos = butter(5, 80, btype="high", fs=sr, output="sos")
    wav = sosfilt(sos, wav).astype(np.float32)

    # Loudness normalization to -20 LUFS
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    if loudness > -70:  # avoid normalizing silence
        wav = pyln.normalize.loudness(wav, loudness, -20.0)

    # Clip to prevent clipping
    wav = np.clip(wav, -1.0, 1.0)

    return wav


def synthesize_coqui(model_path, config_path, text, output_path, speaker=None):
    """Synthesize using Coqui TTS API"""
    from TTS.api import TTS

    # Load model
    tts = TTS(model_path=model_path, config_path=config_path)

    # Move to GPU if available
    if torch.cuda.is_available():
        tts = tts.to("cuda")

    # Synthesize
    if speaker:
        tts.tts_to_file(text=text, file_path=output_path, speaker=speaker)
    else:
        tts.tts_to_file(text=text, file_path=output_path)

    print(f"Saved: {output_path}")


def synthesize_direct(model_path, config_path, text, output_path, speaker=None, apply_postprocess=True):
    """Synthesize by directly loading the model (more control)"""
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.utils.synthesizer import Synthesizer

    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available(),
    )

    wav = synthesizer.tts(text=text, speaker_name=speaker)
    sr = synthesizer.tts_config.audio.sample_rate

    wav_array = np.array(wav)
    if apply_postprocess:
        wav_array = post_process(wav_array, sr)

    # Save
    sf.write(output_path, wav_array, sr)
    print(f"Saved: {output_path}")
    return wav_array


def find_best_model(output_dir):
    """Find the best or latest checkpoint in the output directory"""
    # Look for best_model.pth first
    for root, dirs, files in os.walk(output_dir):
        if "best_model.pth" in files:
            model_path = os.path.join(root, "best_model.pth")
            config_path = os.path.join(root, "config.json")
            if os.path.exists(config_path):
                return model_path, config_path

    # Fall back to latest checkpoint
    checkpoints = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith("checkpoint_") and f.endswith(".pth"):
                checkpoints.append(os.path.join(root, f))

    if checkpoints:
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = checkpoints[0]
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_path):
            return model_path, config_path

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Sinhala TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize (romanized Sinhala)")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument("--speaker", type=str, default="oshadi", help="Speaker name (for multi-speaker)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Auto-find best model in this directory")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable audio post-processing (highpass + loudness normalization)")

    args = parser.parse_args()

    # Find model
    model_path = args.model
    config_path = args.config

    if model_path is None:
        # Auto-detect from output directories
        training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        search_dirs = [
            args.model_dir,
            os.path.join(training_dir, "output_female_only_clean"),
            os.path.join(training_dir, "output"),
            os.path.join(training_dir, "output_female_only"),
        ]

        for d in search_dirs:
            if d and os.path.isdir(d):
                model_path, config_path = find_best_model(d)
                if model_path:
                    break

    if not model_path or not os.path.exists(model_path):
        print("ERROR: No model found. Specify --model or --model-dir")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Text: {args.text}")
    print(f"Speaker: {args.speaker}")
    print()

    synthesize_direct(model_path, config_path, args.text, args.output, args.speaker,
                      apply_postprocess=not args.no_postprocess)
    print("\nDone!")


# Quick batch synthesis
def batch_synthesize(model_path, config_path, texts, output_dir, speaker="oshadi"):
    """Synthesize multiple sentences"""
    os.makedirs(output_dir, exist_ok=True)

    from TTS.utils.synthesizer import Synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available(),
    )

    sr = synthesizer.tts_config.audio.sample_rate
    for i, text in enumerate(texts):
        wav = synthesizer.tts(text=text, speaker_name=speaker)
        wav_array = post_process(np.array(wav), sr)
        out_path = os.path.join(output_dir, f"sample_{i+1:03d}.wav")
        sf.write(out_path, wav_array, sr)
        print(f"[{i+1}/{len(texts)}] {out_path}")

    print(f"\nAll {len(texts)} samples saved to {output_dir}")


if __name__ == "__main__":
    main()
