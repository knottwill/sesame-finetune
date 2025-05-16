from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
MIMI_SAMPLE_RATE = int(os.getenv("MIMI_SAMPLE_RATE", 24_000))
BACKBONE_FLAVOR = os.getenv("BACKBONE_FLAVOR", "llama-1B")
DECODER_FLAVOR = os.getenv("DECODER_FLAVOR", "llama-100M")
TEXT_VOCAB_SIZE = int(os.getenv("TEXT_VOCAB_SIZE", 128256))
AUDIO_VOCAB_SIZE = int(os.getenv("AUDIO_VOCAB_SIZE", 2051))
AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS", 32))