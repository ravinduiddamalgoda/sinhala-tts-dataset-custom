"""
Sinhala TTS - FastAPI Server
============================
Serves the trained VITS model via a REST API.
Supports both multi-speaker and single-speaker models.

Endpoints:
  POST /synthesize       - JSON body: {"text": "...", "speaker": "oshadi"}  → returns WAV audio
  GET  /synthesize?text=...&speaker=oshadi                                  → returns WAV audio
  GET  /speakers         - List available speakers
  GET  /health           - Health check
  GET  /                 - Interactive web UI for testing

Usage:
  python tts_server.py                          # auto-detect model
  python tts_server.py --model-dir output/      # specify output dir
  python tts_server.py --model best_model.pth --config config.json

The server binds to 0.0.0.0:8000 by default (accessible via public IP / ngrok).
"""

import os
import sys
import io
import argparse
import time
import logging
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import find_best_model, post_process

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("sinhala-tts")

# ── Globals (loaded once at startup) ─────────────────────────────────
synthesizer = None
is_multi_speaker = False
speaker_names = []
sample_rate = 22050


# ── Pydantic models ─────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    speed: Optional[float] = 1.0


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sinhala TTS API",
    description="Sinhala Text-to-Speech API powered by VITS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _synthesize(text: str, speaker: str = None) -> bytes:
    """Run TTS and return WAV bytes."""
    global synthesizer, is_multi_speaker, sample_rate

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    speaker_name = None
    if is_multi_speaker:
        speaker_name = speaker or "oshadi"
        if speaker_name not in speaker_names:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker '{speaker_name}'. Available: {speaker_names}",
            )

    t0 = time.time()
    wav = synthesizer.tts(text=text, speaker_name=speaker_name)
    elapsed = time.time() - t0
    logger.info(f"Synthesized {len(text)} chars in {elapsed:.2f}s | speaker={speaker_name}")

    # Post-process and convert to WAV bytes
    wav_array = np.array(wav, dtype=np.float32)
    wav_array = post_process(wav_array, sample_rate)
    buf = io.BytesIO()
    sf.write(buf, wav_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": synthesizer is not None,
        "multi_speaker": is_multi_speaker,
        "speakers": speaker_names,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/speakers")
def get_speakers():
    return {"speakers": speaker_names, "default": "oshadi" if speaker_names else None}


@app.post("/synthesize")
def synthesize_post(req: SynthesizeRequest):
    buf = _synthesize(req.text, req.speaker)
    return StreamingResponse(buf, media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output.wav"
    })


@app.get("/synthesize")
def synthesize_get(
    text: str = Query(..., description="Romanized Sinhala text"),
    speaker: Optional[str] = Query(None, description="Speaker name"),
):
    buf = _synthesize(text, speaker)
    return StreamingResponse(buf, media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output.wav"
    })


@app.get("/", response_class=HTMLResponse)
def web_ui():
    """Simple web UI for interactive testing."""
    speakers_options = "".join(
        f'<option value="{s}" {"selected" if s == "oshadi" else ""}>{s}</option>'
        for s in speaker_names
    ) if speaker_names else '<option value="">default</option>'

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sinhala TTS</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
        .container {{ max-width: 600px; width: 90%; padding: 2rem; }}
        h1 {{ text-align: center; margin-bottom: 0.5rem; font-size: 1.8rem; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }}
        label {{ display: block; margin-bottom: 0.3rem; font-weight: 600; font-size: 0.85rem; color: #94a3b8; }}
        textarea {{ width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 1rem; resize: vertical; min-height: 100px; margin-bottom: 1rem; }}
        textarea:focus {{ outline: none; border-color: #3b82f6; }}
        select {{ width: 100%; padding: 0.6rem; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; margin-bottom: 1rem; }}
        button {{ width: 100%; padding: 0.75rem; border-radius: 8px; border: none; background: #3b82f6; color: white; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }}
        button:hover {{ background: #2563eb; }}
        button:disabled {{ background: #475569; cursor: not-allowed; }}
        .result {{ margin-top: 1rem; text-align: center; }}
        audio {{ width: 100%; margin-top: 0.5rem; }}
        .status {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; text-align: center; }}
        .examples {{ margin-top: 1rem; }}
        .examples span {{ display: inline-block; background: #334155; padding: 0.3rem 0.6rem; border-radius: 6px; margin: 0.2rem; font-size: 0.8rem; cursor: pointer; transition: background 0.2s; }}
        .examples span:hover {{ background: #475569; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sinhala TTS</h1>
        <p class="subtitle">Romanized Sinhala Text-to-Speech</p>
        <div class="card">
            <label for="text">Text (romanized Sinhala)</label>
            <textarea id="text" placeholder="e.g. ayubōvan, mama ōśadī."></textarea>

            <label for="speaker">Speaker</label>
            <select id="speaker">{speakers_options}</select>

            <button id="btn" onclick="synthesize()">Synthesize</button>

            <div class="result" id="result" style="display:none;">
                <audio id="audio" controls></audio>
            </div>
            <div class="status" id="status"></div>

            <div class="examples">
                <label>Examples (click to use):</label>
                <span onclick="useExample(this)">ayubōvan, mama ōśadī.</span>
                <span onclick="useExample(this)">śrī laṁkāvē rājya bhāşāva siṁhala ya.</span>
                <span onclick="useExample(this)">mē vanayen kæḷǣ rakşitayak.</span>
                <span onclick="useExample(this)">amma, tātta mallī saha naṁgilā dedena apē pavulē sāmājikayan.</span>
            </div>
        </div>
    </div>
    <script>
        function useExample(el) {{
            document.getElementById('text').value = el.textContent;
        }}
        async function synthesize() {{
            const text = document.getElementById('text').value.trim();
            if (!text) return;
            const btn = document.getElementById('btn');
            const status = document.getElementById('status');
            btn.disabled = true;
            btn.textContent = 'Synthesizing...';
            status.textContent = '';
            const t0 = performance.now();
            try {{
                const resp = await fetch('/synthesize', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        text: text,
                        speaker: document.getElementById('speaker').value || null
                    }})
                }});
                if (!resp.ok) {{
                    const err = await resp.json();
                    throw new Error(err.detail || 'Synthesis failed');
                }}
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const audio = document.getElementById('audio');
                audio.src = url;
                document.getElementById('result').style.display = 'block';
                audio.play();
                const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
                status.textContent = `Done in ${{elapsed}}s`;
            }} catch (e) {{
                status.textContent = 'Error: ' + e.message;
            }} finally {{
                btn.disabled = false;
                btn.textContent = 'Synthesize';
            }}
        }}
        document.getElementById('text').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') synthesize();
        }});
    </script>
</body>
</html>
"""


# ── Model loading ────────────────────────────────────────────────────

def load_model(model_path: str, config_path: str):
    """Load the TTS model into the global synthesizer."""
    global synthesizer, is_multi_speaker, speaker_names, sample_rate

    from TTS.utils.synthesizer import Synthesizer as TTSSynthesizer

    logger.info(f"Loading model: {model_path}")
    logger.info(f"Config: {config_path}")

    synthesizer = TTSSynthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available(),
    )

    sample_rate = synthesizer.tts_config.audio.sample_rate

    # Detect speakers
    if hasattr(synthesizer.tts_model, "num_speakers") and synthesizer.tts_model.num_speakers > 1:
        is_multi_speaker = True
        # Try to get speaker names from the model's speaker manager
        if hasattr(synthesizer.tts_model, "speaker_manager") and synthesizer.tts_model.speaker_manager:
            sm = synthesizer.tts_model.speaker_manager
            if hasattr(sm, "name_to_id"):
                speaker_names = list(sm.name_to_id.keys())
            elif hasattr(sm, "speaker_names"):
                speaker_names = list(sm.speaker_names)
        if not speaker_names:
            speaker_names = ["oshadi", "mettananda"]  # fallback from training config
    else:
        is_multi_speaker = False
        speaker_names = []

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Model loaded on {device} | Multi-speaker: {is_multi_speaker} | Speakers: {speaker_names}")
    logger.info(f"Sample rate: {sample_rate}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sinhala TTS FastAPI Server")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--model-dir", type=str, default=None, help="Auto-find best model in this directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    model_path = args.model
    config_path = args.config

    # Auto-detect model
    if model_path is None:
        training_dir = os.path.dirname(os.path.abspath(__file__))
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
        logger.error("No model found! Specify --model and --config, or --model-dir")
        logger.error("Expected locations: training/output/*/best_model.pth")
        sys.exit(1)

    load_model(model_path, config_path)

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Web UI: http://{args.host}:{args.port}/")
    logger.info(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
