from typing import List
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


class TokenizedDataset(torch.utils.data.Dataset):
    """Tokenized Dataset for the CSM model

    Args:
        audio_tokens: audio token ids (each token is a list of codebook ids)
        text_tokens: text token ids (each token is a single id)
    """

    def __init__(self, audio_tokens: List[List[List[int]]], text_tokens: List[List[int]]):
        self.audio_tokens = audio_tokens
        self.text_tokens = text_tokens

    def __len__(self):
        return len(self.audio_tokens)

    def __getitem__(self, index: int):
        return {"audio": self.audio_tokens[index], "text": self.text_tokens[index]}


def collate_fn(batch: List[dict]):
    """Collate function for the TokenizedDataset"""
    tokens, tokens_mask = [], []
    n_codebooks = 32
    for item in batch:
        audio_tokens = torch.tensor(item["audio"])
        text_tokens = torch.tensor(item["text"])

        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)  # [n_codebooks, audio_seq_len+1]

        # add extra dimension for text ids
        audio_frame = torch.zeros(audio_tokens.size(1), n_codebooks + 1).long()  # [audio_seq_len+1, n_codebooks+1]
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), n_codebooks + 1).bool()  # [audio_seq_len+1, n_codebooks+1]
        audio_frame_mask[:, :-1] = True

        text_frame = torch.zeros(len(text_tokens), n_codebooks + 1).long()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask = torch.zeros(len(text_tokens), n_codebooks + 1).bool()
        text_frame_mask[:, -1] = True

        tokens.append(torch.cat([text_frame, audio_frame], dim=0))
        tokens_mask.append(torch.cat([text_frame_mask, audio_frame_mask], dim=0))

    tokens = pad_sequence(tokens, batch_first=True)
    tokens_mask = pad_sequence(tokens_mask, batch_first=True, padding_value=False)

    return tokens, tokens_mask


class CSMSampler(torch.utils.data.sampler.Sampler):
    """Sampler that groups samples of similar lengths to minimize padding in batches."""

    def __init__(
        self, lengths: List[int], batch_size: int, shuffle: bool = True, is_infinite: bool = True, random_seed: int = 42
    ):
        """
        lengths:      List of sequence lengths for each sample
        is_infinite:  Whether to repeat the dataset infinitely
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_infinite = is_infinite
        self.random_seed = random_seed
        self.local_step = 0
        self.bins = self._create_bins(lengths, batch_size)

    def _create_bins(self, lengths: List[int], batch_size: int) -> List[List[int]]:
        """Group samples of similar lengths into bins"""
        indices_with_lengths = [(i, length) for i, length in enumerate(lengths)]
        indices_with_lengths.sort(key=lambda x: x[1])

        # Group into bins
        bins = []
        current_bin = []
        for idx, length in indices_with_lengths:
            if len(current_bin) >= batch_size:
                bins.append(current_bin)
                current_bin = []
            current_bin.append(idx)

        if current_bin:
            bins.append(current_bin)

        return bins

    def _shuffle_bins(self, epoch: int):
        """
        epoch: Current epoch number for deterministic shuffling
        """
        rng = np.random.RandomState(epoch + self.random_seed)
        rng.shuffle(self.bins)  # shuffle bins
        for i in range(len(self.bins)):  # shuffle samples in each bin
            self.bins[i] = [self.bins[i][j] for j in rng.permutation(len(self.bins[i]))]

    def __iter__(self):
        epoch = 0
        while True:
            if self.shuffle:
                self._shuffle_bins(epoch)

            for bin_indices in self.bins:
                self.local_step += 1
                yield bin_indices

            if not self.is_infinite:
                break

            epoch += 1

    def __len__(self):
        return len(self.bins)


def create_dataloaders(all_tokens: dict, batch_size: int, infinite_train: bool = True):
    """Create dataloaders for the CSM model

    all_tokens = {
        "audio_tokens_train": audio_tokens_train,
        "text_tokens_train": text_tokens_train,
        "audio_tokens_val": audio_tokens_val,
        "text_tokens_val": text_tokens_val,
    }
    """
    trainset = TokenizedDataset(all_tokens["audio_tokens_train"], all_tokens["text_tokens_train"])
    valset = TokenizedDataset(all_tokens["audio_tokens_val"], all_tokens["text_tokens_val"])

    trainsampler = CSMSampler(
        lengths=[len(tokens) for tokens in all_tokens["audio_tokens_train"]],
        batch_size=batch_size,
        is_infinite=infinite_train,
        shuffle=True,
    )

    valsampler = CSMSampler(
        lengths=[len(tokens) for tokens in all_tokens["audio_tokens_val"]],
        batch_size=batch_size,
        is_infinite=False,
        shuffle=False,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_sampler=trainsampler, num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    valloader = torch.utils.data.DataLoader(
        valset, batch_sampler=valsampler, num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    return trainloader, valloader
