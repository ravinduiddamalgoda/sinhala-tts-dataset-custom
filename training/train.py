"""
Sinhala TTS - VITS Multi-Speaker Training Script
=================================================
Uses Coqui TTS to train a VITS model on the Sinhala TTS dataset.
Optimized for 2x RTX 5060 Ti (16GB VRAM each).
"""

import os
import sys

# ---- CONFIGURATION ----
# Set these paths according to your setup
DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(DATASET_ROOT, "training")

# Use multi-speaker dataset (recommended - better quality with more data)
DATASET_PATH = os.path.join(TRAINING_DIR, "dataset", "multispeaker")
OUTPUT_PATH = os.path.join(TRAINING_DIR, "output")

# Training parameters optimized for 2x RTX 5060 Ti
BATCH_SIZE = 32         # Per GPU (total effective = 64 with 2 GPUs)
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 1000       # VITS converges around 200-500K steps
LEARNING_RATE = 0.0002
NUM_WORKERS = 8

# Audio parameters (matching dataset: 22050Hz)
SAMPLE_RATE = 22050

# ---- END CONFIGURATION ----

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainerArgs

# Define the character set from the dataset (romanized Sinhala)
# Characters from the dataset README
CHARACTERS = " !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ"

def main():
    print("=" * 60)
    print("  Sinhala VITS Multi-Speaker Training")
    print("=" * 60)

    # Verify dataset exists
    train_meta = os.path.join(DATASET_PATH, "metadata_train.csv")
    val_meta = os.path.join(DATASET_PATH, "metadata_val.csv")

    if not os.path.exists(train_meta):
        print(f"\nERROR: Training metadata not found: {train_meta}")
        print("Run prepare_data.py first!")
        sys.exit(1)

    # Count entries
    with open(train_meta, "r", encoding="utf-8") as f:
        train_count = sum(1 for line in f if line.strip())
    with open(val_meta, "r", encoding="utf-8") as f:
        val_count = sum(1 for line in f if line.strip())

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Batch size: {BATCH_SIZE} per GPU")

    # Audio config matching the dataset
    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    # Character config for romanized Sinhala
    character_config = CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        characters=CHARACTERS,
        punctuations="!'(),-.:;=?",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
    )

    # VITS model configuration
    config = VitsConfig(
        output_path=OUTPUT_PATH,
        run_name="sinhala-vits-multispeaker",

        # Audio
        audio=audio_config,

        # Model architecture
        use_speaker_embedding=True,
        num_speakers=2,

        # Characters
        characters=character_config,
        text_cleaner=None,
        use_phonemes=False,

        # Training parameters
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=NUM_WORKERS,
        num_eval_loader_workers=4,
        epochs=NUM_EPOCHS,

        # Optimizer
        lr_gen=LEARNING_RATE,
        lr_disc=LEARNING_RATE,

        # Scheduler
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},

        # Logging & Checkpointing
        print_step=50,
        plot_step=100,
        print_eval=True,
        mixed_precision=True,  # Use FP16 for faster training
        save_step=5000,
        save_n_checkpoints=3,
        save_best_after=10000,

        # Evaluation
        run_eval=True,
        test_delay_epochs=5,

        # Dataset
        datasets=[
            {
                "formatter": "ljspeech",
                "meta_file_train": "metadata_train.csv",
                "meta_file_val": "metadata_val.csv",
                "path": DATASET_PATH,
                "language": "si",
            }
        ],

        # Test sentences for monitoring progress (romanized Sinhala)
        test_sentences=[
            ["ayubovan, mama ōśadī.", "oshadi", None, None],
            ["mē siṁhala parigaṇaka handa ha'ḍunaganīma sanḍahā vū parikşaṇayak.", "oshadi", None, None],
            ["āyuşmat bakkula teraṇuvō", "oshadi", None, None],
            ["śrī laṁkāvē rājya bhāşāva siṁhala ya.", "oshadi", None, None],
            ["apē raṭē kāntāvanṭa dinā gatu yutu ayitivāsikam ræsak tibenavā.", "oshadi", None, None],
        ],

        # Cudnn benchmark for consistent GPU performance
        cudnn_benchmark=True,
    )

    # Load dataset samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_batch_size * 10,
        eval_split_size=0.05,
    )

    print(f"\nLoaded {len(train_samples)} training / {len(eval_samples)} eval samples")

    # Initialize model
    model = Vits.init_from_config(config)

    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            gpu=0,
        ),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting training...")
    print("Monitor with: tensorboard --logdir", OUTPUT_PATH)
    print("")

    # Train!
    trainer.fit()


if __name__ == "__main__":
    main()
