"""
Script to pre-tokenize training/validation data for Sesame finetuning and save incrementally in HDF5.

Usage:
python pretokenize.py --train_data /path/to/train.json --val_data /path/to/val.json --output /path/to/output/data.hdf5
"""

import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import torch
import torchaudio
import h5py
import numpy as np
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from utils import load_tokenizers, MIMI_SAMPLE_RATE


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="data.hdf5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=100, help="Save every N samples")
    args = parser.parse_args(arg_string.split() if arg_string else None)
    return args


def load_metadata(data_path: Path | str) -> pd.DataFrame:
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


def append_to_hdf5(file_path, split, audio_tokens_batch, text_tokens_batch, compression="gzip"):
    """
    Append audio and text tokens to HDF5 file.
    """
    with h5py.File(file_path, "a") as f:
        grp = f.require_group(split)

        def prepare_ds(name, shape, dtype, is_vlen=False):
            """
            Create a dataset if it doesn't exist, otherwise return the existing one.
            """
            if name not in grp:
                maxshape = (None,) + shape # None allows for dynamic resizing
                return grp.create_dataset(name, shape=(0,) + shape, maxshape=maxshape, dtype=dtype, compression=compression)
            return grp[name]

        audio_shape = np.array(audio_tokens_batch[0], dtype=np.int32).shape
        audio_ds = prepare_ds("audio", audio_shape, np.int32)
        text_ds = grp.get("text") or grp.create_dataset("text", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))
        length_ds = grp.get("length") or grp.create_dataset("length", shape=(0,), maxshape=(None,), dtype=np.int32)

        n = len(audio_tokens_batch)
        audio_ds.resize((audio_ds.shape[0] + n), axis=0)
        text_ds.resize((text_ds.shape[0] + n), axis=0)
        length_ds.resize((length_ds.shape[0] + n), axis=0)

        for i in range(n):
            audio_array = np.array(audio_tokens_batch[i], dtype=np.int32)
            text_array = np.array(text_tokens_batch[i], dtype=np.int32)
            total_len = audio_array.shape[1] + len(text_array) + 1  # +1 for EOS frame

            audio_ds[-n + i] = audio_array
            text_ds[-n + i] = text_array
            length_ds[-n + i] = total_len


def tokenize_and_store(data_path, output_path, split, audio_tokenizer, text_tokenizer, device, save_every=100):
    df = load_metadata(data_path)
    audio_tokens_batch, text_tokens_batch = [], []

    print(f"Processing {split} split: {len(df)} samples")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        # Handle optional timestamps
        if "start" in row and "end" in row:
            sr = torchaudio.info(row["path"]).sample_rate
            frame_offset = int(row["start"] * sr)
            num_frames = int((row["end"] - row["start"]) * sr)
        else:
            frame_offset = 0
            num_frames = -1

        # Load and resample audio
        waveform, sr = torchaudio.load(row["path"], frame_offset=frame_offset, num_frames=num_frames)
        waveform = torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sr, new_freq=MIMI_SAMPLE_RATE)
        waveform = waveform.unsqueeze(0).unsqueeze(0).to(device)

        # Tokenize
        audio_tokens = audio_tokenizer.encode(waveform)[0].tolist()
        speaker = row.get("speaker", 999)
        text = f"[{speaker}]{row['text']}"
        text_tokens = text_tokenizer.encode(text)

        # Accumulate batch
        audio_tokens_batch.append(audio_tokens)
        text_tokens_batch.append(text_tokens)

        if len(audio_tokens_batch) >= save_every:
            append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)
            audio_tokens_batch, text_tokens_batch = [], []

    # Final flush
    if audio_tokens_batch:
        append_to_hdf5(output_path, split, audio_tokens_batch, text_tokens_batch)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    tokenize_and_store(
        args.train_data, output_path=args.output, split="train",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every
    )

    tokenize_and_store(
        args.val_data, output_path=args.output, split="val",
        audio_tokenizer=audio_tokenizer, text_tokenizer=text_tokenizer,
        device=device, save_every=args.save_every
    )

    print(f"\n✅ Done. Tokenized data saved to: {args.output}")
