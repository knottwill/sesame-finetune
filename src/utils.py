import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from pathlib import Path
from typing import Union
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from . import MIMI_SAMPLE_RATE, TEXT_VOCAB_SIZE, AUDIO_VOCAB_SIZE, AUDIO_NUM_CODEBOOKS
from .generator import Generator
from .models import Model, ModelArgs, _create_causal_mask
from .watermarking import load_watermarker


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer


def load_mimi(device: Union[str, torch.device]):
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(AUDIO_NUM_CODEBOOKS)
    return mimi


def load_tokenizers(device: Union[str, torch.device]):
    """Load text and audio tokenizers."""
    text_tokenizer = load_llama3_tokenizer()
    audio_tokenizer = load_mimi(device=device)
    return text_tokenizer, audio_tokenizer


def load_model(
        model_name_or_checkpoint_path: Union[str, Path] = None,
        device: Union[str, torch.device] = 'cuda',
        decoder_loss_weight: float = 0.5
    ) -> Model:
    """Load model, add forward method, and move to device.
    
    Args:
        model_name_or_checkpoint_path: Name or path of pretrained model or checkpoint.
        device: Device to move the model to.
        decoder_loss_weight: Decoder loss weight.
    """
    if model_name_or_checkpoint_path is None or model_name_or_checkpoint_path.endswith(".pt"):
        # Training from scratch or using local checkpoint
        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=TEXT_VOCAB_SIZE,
            audio_vocab_size=AUDIO_VOCAB_SIZE,
            audio_num_codebooks=AUDIO_NUM_CODEBOOKS,
            decoder_loss_weight=decoder_loss_weight
        )
        model = Model(config)

        if model_name_or_checkpoint_path:
            state_dict = torch.load(model_name_or_checkpoint_path)['model']
            model.load_state_dict(state_dict)
        else:
            model = init_weights(model)
    else: 
        # Huggingface model name or local path
        model = Model.from_pretrained(model_name_or_checkpoint_path)
        
    return model.to(device=device)


class WarmupDecayLR(LambdaLR):
    """
    Learning rate scheduler with a linear warmup and specificable decay.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, decay_type: str = "linear"):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        else:
            if self.decay_type == "linear":
                return (self.total_steps - step) / (self.total_steps - self.warmup_steps)
            elif self.decay_type == "constant":
                return 1.0
            elif self.decay_type == "exponential":
                return 0.1 ** ((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            elif self.decay_type == "cosine":
                return 0.5 * (1 + torch.cos(torch.pi * torch.tensor((step - self.warmup_steps) / (self.total_steps - self.warmup_steps))))
            else:
                raise ValueError(f"Invalid decay type: {self.decay_type}")


def init_weights(model: nn.Module):
    """
    Initialize the weights of the model.
    - Xavier uniform initialization for linear layers
    - Normal initialization for embeddings
    - Xavier uniform initialization for parameters
    """

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data)

    model.apply(_init_weights)

    # Special handling for audio_head because it's nn.Parameter directly
    nn.init.xavier_uniform_(model.audio_head)

    return model


def reset_caches(model: Model):
    """Reset the caches of the model (used after each generation)."""
    model.reset_caches()
    for module in model.modules():
        if hasattr(module, "cache_enabled"):
            module.cache_enabled = False
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def generate_audio(model, audio_tokenizer, text_tokenizer, watermarker, text, speaker_id, device, use_amp=True, max_audio_length_ms=10_000):
    """Generate audio from text."""
    model.eval()
    generator = Generator(model, audio_tokenizer, text_tokenizer, watermarker)
    
    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
        )
        audio = audio.squeeze().cpu().numpy()
    
    reset_caches(model)
    return audio


def validate(model, valloader, device, use_amp=True):
    """Validate the model on the validation set."""
    model.eval()
    val_losses = []
    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        for val_tokens, val_tokens_mask in valloader:
            val_tokens = val_tokens.to(device)
            val_tokens_mask = val_tokens_mask.to(device)
            val_loss = model(val_tokens, val_tokens_mask).item()
            val_losses.append(val_loss)
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss
