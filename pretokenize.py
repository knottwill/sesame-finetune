"""
Script to pre-tokenize the training and validation data for the sesame finetune (fastest on a GPU).

Usage:
python pretokenize.py --train_data /path/to/train/metadata.json --val_data /path/to/val/metadata.json --output /path/to/output/tokens.pkl
"""

import argparse
import pickle
import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from utils import load_tokenizers, MIMI_SAMPLE_RATE


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, help="Path to the training metadata", required=True)
    parser.add_argument("--val_data", type=Path, help="Path to the validation metadata", required=True)
    parser.add_argument("--output", type=Path, default="./data/tokens.pkl", help="Path to save the computed tokens", required=True)
    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def load_metadata(data_path: Path | str):
    """
    Metadata should have the following columns:
    - path: Path to the audio wav file.
    - text: Text transcription of the audio.
    - (Optional) start: Start time of the transcription in the wav file.
    - (Optional) end: End time of the transcription in the wav file.
    - (Optional) speaker: Speaker id.

    Supported file formats: json, csv, sql, parquet, pkl. Feel free to add more.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    if data_path.suffix == ".json":
        return pd.read_json(data_path)
    elif data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    elif data_path.suffix == ".sql":
        return pd.read_sql_query("SELECT * FROM data", sqlite3.connect(data_path))
    elif data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        return pd.read_pickle(data_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {data_path}")


def get_tokens(
    data_path: Path, audio_tokenizer: nn.Module, text_tokenizer: PreTrainedTokenizerFast, device: torch.device
) -> Tuple[List[List[int]], List[List[int]]]:
    """Pre-tokenize the data"""
    df = load_metadata(data_path)
    audio_tokens, text_tokens = [], []
    for _, data_point in tqdm(df.iterrows()):
        if "start" and "end" in data_point:
            sample_rate = torchaudio.info(data_point["path"]).sample_rate
            frame_offset = int(data_point["start"] * sample_rate)
            num_frames = int((data_point["end"] - data_point["start"]) * sample_rate)
        else:
            frame_offset = 0
            num_frames = -1

        # load audio
        audio_tensor, sample_rate = torchaudio.load(
            data_point["path"], frame_offset=frame_offset, num_frames=num_frames
        )
        audio_tensor = torchaudio.functional.resample(audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=MIMI_SAMPLE_RATE)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)

        # tokenize audio
        audio_tokens.append(audio_tokenizer.encode(audio_tensor)[0].tolist())

        # tokenize text
        text = f"[{data_point['speaker'] if 'speaker' in data_point else 999}]" + data_point["text"]
        text_tokens.append(text_tokenizer.encode(text))

    return audio_tokens, text_tokens


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    print(f"Tokenizing training data from {args.train_data}")
    audio_tokens_train, text_tokens_train = get_tokens(args.train_data, audio_tokenizer, text_tokenizer, device)
    print(f"Tokenizing validation data from {args.val_data}")
    audio_tokens_val, text_tokens_val = get_tokens(args.val_data, audio_tokenizer, text_tokenizer, device)
    all_tokens = {
        "audio_tokens_train": audio_tokens_train,
        "text_tokens_train": text_tokens_train,
        "audio_tokens_val": audio_tokens_val,
        "text_tokens_val": text_tokens_val,
    }

    with open(args.output, "wb") as f:
        pickle.dump(all_tokens, f)
