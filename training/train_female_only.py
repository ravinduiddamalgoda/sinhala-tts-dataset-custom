"""
Sinhala TTS - Female-Only VITS Training Script
===============================================
Alternative: Train VITS on only female (oshadi) data.
Use this if you want a simpler single-speaker model.
~985 clips / ~2 hours of female speech.
"""

import os
import sys

DATASET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(DATASET_ROOT, "training")
DATASET_PATH = os.path.join(TRAINING_DIR, "dataset", "female_only")
OUTPUT_PATH = os.path.join(TRAINING_DIR, "output_female_only")

BATCH_SIZE = 24          # Smaller because less data
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 2000        # More epochs needed for small dataset
LEARNING_RATE = 0.0002
SAMPLE_RATE = 22050
NUM_WORKERS = 4

CHARACTERS = " !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ"

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.trainer import Trainer, TrainerArgs


def main():
    print("=" * 60)
    print("  Sinhala VITS Female-Only Training")
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
        mel_fmax=None,
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
        run_name="sinhala-vits-female",

        audio=audio_config,

        # Single speaker
        use_speaker_embedding=False,
        num_speakers=0,

        characters=character_config,
        text_cleaner=None,
        use_phonemes=False,

        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=NUM_WORKERS,
        num_eval_loader_workers=2,
        epochs=NUM_EPOCHS,

        lr_gen=LEARNING_RATE,
        lr_disc=LEARNING_RATE,
        lr_scheduler_gen="ExponentialLR",
        lr_scheduler_gen_params={"gamma": 0.999875, "last_epoch": -1},
        lr_scheduler_disc="ExponentialLR",
        lr_scheduler_disc_params={"gamma": 0.999875, "last_epoch": -1},

        print_step=25,
        plot_step=100,
        print_eval=True,
        mixed_precision=True,
        save_step=2000,
        save_n_checkpoints=3,
        save_best_after=5000,

        run_eval=True,
        test_delay_epochs=3,

        datasets=[
            {
                "formatter": "ljspeech",
                "meta_file_train": "metadata_train.csv",
                "meta_file_val": "metadata_val.csv",
                "path": DATASET_PATH,
                "language": "si",
            }
        ],

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
    trainer.fit()


if __name__ == "__main__":
    main()
