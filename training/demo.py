"""
Sinhala TTS - Demo Web Interface
================================
Simple Gradio interface for demonstrating the trained model.
Run after training to demo the TTS system.
"""

import os
import sys
import torch
import numpy as np
import tempfile

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import find_best_model


def create_demo():
    try:
        import gradio as gr
    except ImportError:
        print("Installing gradio...")
        os.system("pip install gradio")
        import gradio as gr

    from TTS.utils.synthesizer import Synthesizer

    # Find model
    training_dir = os.path.dirname(os.path.abspath(__file__))
    model_path, config_path = None, None

    for d in ["output", "output_female_only"]:
        search_dir = os.path.join(training_dir, d)
        if os.path.isdir(search_dir):
            model_path, config_path = find_best_model(search_dir)
            if model_path:
                break

    if not model_path:
        print("ERROR: No trained model found!")
        print("Train a model first, then run this demo.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available(),
    )

    # Check if multi-speaker
    is_multi_speaker = hasattr(synthesizer.tts_model, 'num_speakers') and synthesizer.tts_model.num_speakers > 1

    def synthesize(text, speaker="oshadi"):
        if not text.strip():
            return None
        speaker_name = speaker if is_multi_speaker else None
        wav = synthesizer.tts(text=text, speaker_name=speaker_name)
        sample_rate = synthesizer.tts_config.audio.sample_rate
        return (sample_rate, np.array(wav, dtype=np.float32))

    # Example sentences (romanized Sinhala)
    examples = [
        ["ayubōvan, mama ōśadī."],
        ["siṁhala parigaṇaka handa ha'ḍunaganīma sanḍahā vū parikşaṇayak."],
        ["apē raṭē kāntāvanṭa dinā gatu yutu ayitivāsikam ræsak tibenavā."],
        ["śrī laṁkāvē rājya bhāşāva siṁhala ya."],
        ["mē vanayen kæḷǣ rakşitayak. ēvāyē dara kapanna tahanam."],
        ["amma, tātta mallī saha naṁgilā dedena apē pavulē sāmājikayan."],
        ["ada ekī pāsala hæ'ḍinvennē ruvanvælla rājakīya vidyālaya yanuveni."],
    ]

    # Build interface
    with gr.Blocks(title="Sinhala TTS Demo") as demo:
        gr.Markdown("# 🇱🇰 Sinhala Text-to-Speech Demo")
        gr.Markdown("Enter romanized Sinhala text to generate speech.")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Romanized Sinhala Text",
                    placeholder="Type or select an example...",
                    lines=3,
                )
                if is_multi_speaker:
                    speaker_dropdown = gr.Dropdown(
                        choices=["oshadi", "mettananda"],
                        value="oshadi",
                        label="Speaker",
                    )
                synthesize_btn = gr.Button("🔊 Synthesize", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech", type="numpy")

        gr.Examples(
            examples=examples,
            inputs=[text_input],
        )

        if is_multi_speaker:
            synthesize_btn.click(
                fn=synthesize,
                inputs=[text_input, speaker_dropdown],
                outputs=[audio_output],
            )
        else:
            synthesize_btn.click(
                fn=lambda text: synthesize(text),
                inputs=[text_input],
                outputs=[audio_output],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates a public link
    )
