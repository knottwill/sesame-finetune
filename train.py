import argparse
import os
import pickle
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import wandb
from aladdin.tts.sesame_finetune.dataloaders import create_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("--tokenized_data", type=str, required=True, help="Path to the pre-tokenized data")
parser.add_argument("--output_dir", type=Path, default="./exp", help="Path to save the model")
parser.add_argument(
    "--model",
    type=str,
    default="sesame/csm-1b",
    help="Option to specify local path if you have already downloaded the model",
)
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--wandb_project", type=str, default="csm-finetuning", help="Name of the project")
parser.add_argument("--wandb_name", type=str, default=None, help="Name of the run")
parser.add_argument("--wandb_reinit", type=bool, default=True, help="Whether to reinitialize the run")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--total_steps", type=int, default=5000)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--val_every", type=int, default=100)
parser.add_argument("--gen_every", type=int, default=100)
parser.add_argument(
    "--gen_sentence",
    type=str,
    default="Bird law in this country is not governed by reason.",
    help="Sentence for model to generate (should be the same language as the tokenized data)",
)
parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")
args = parser.parse_args()

# imports from official sesame csm repo
try:
    import sys

    sys.path.append(args.csm_path)
    from generator import Generator, load_llama3_tokenizer, load_watermarker
    from models import Model, _create_causal_mask
except ImportError:
    raise ImportError(
        "Please clone https://github.com/SesameAILabs/csm.git to use this script and then specify the --csm_path flag"
    )


def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor):
    """
    Forward pass for Sesame's CSM model.
    This will be added to the model with `model.forward = types.MethodType(forward, model)`

    TODO: Add a better description here

    Args:
        tokens: (batch_size, seq_len, audio_num_codebooks+1)
        tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
    """
    dtype = next(self.parameters()).dtype
    bsz, seq_len, _ = tokens.size()
    device = tokens.device

    embeds = self._embed_tokens(tokens)

    # retain just non-padding embeddings
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)

    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]  # [bsz, seq_len]
    backbone_attn_mask = _create_causal_mask(seq_len, device)  # [seq_len, seq_len]
    padding_3d = padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1)  # [bsz, seq_len, seq_len]
    backbone_attn_mask = backbone_attn_mask.unsqueeze(0) * padding_3d
    backbone_attn_mask = backbone_attn_mask | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1)
    input_pos = torch.arange(0, seq_len).unsqueeze(0).expand(bsz, seq_len).long().to(device)
    h = self.backbone(h, input_pos=input_pos, mask=backbone_attn_mask).to(dtype=dtype)

    audio_mask = tokens_mask[:, :, 0]  # [bsz, seq_len]
    target_tokens = tokens[audio_mask][:, :-1]  # [audio_len, audio_num_codebooks]
    c_embeds = embeds[:, :, :-1, :][audio_mask]  # [audio_len, audio_num_codebooks, embed_dim]
    c_pos = input_pos[audio_mask]  # [audio_len]

    audio_mask = torch.roll(audio_mask, -1, 1)
    audio_h = h[audio_mask]  # [audio_len, embed_dim]

    c0_logits = self.codebook0_head(audio_h)  # [audio_len, audio_vocab_size]
    c0_target = target_tokens[:, 0]  # [audio_len]
    c0_loss = F.cross_entropy(c0_logits, c0_target)

    # compute amortization
    indices = torch.randperm(c_embeds.size(0))[: c_embeds.size(0) // 16]
    c_embeds = c_embeds[indices][:, :-1, :]  # [audio_len//16, audio_num_codebooks-1, embed_dim]
    audio_h = audio_h[indices]  # [audio_len//16, embed_dim]
    target_tokens = target_tokens[indices][:, 1:]  # [audio_len//16, audio_num_codebooks-1]

    decoder_embeds = torch.cat(
        [audio_h.unsqueeze(1), c_embeds], dim=1
    )  # [audio_len//16, audio_num_codebooks, embed_dim]
    N, audio_num_codebooks, _ = decoder_embeds.size()
    c_pos = torch.arange(0, audio_num_codebooks).unsqueeze(0).expand(N, audio_num_codebooks).long().to(device)

    decoder_causal_mask = _create_causal_mask(decoder_embeds.size(1), device).expand(N, -1, -1)
    decoder_h = self.decoder(self.projection(decoder_embeds), input_pos=c_pos, mask=decoder_causal_mask).to(dtype=dtype)
    c_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)

    c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_tokens.reshape(-1))

    loss = c0_loss + c_loss
    return loss


def custom_generator_init(self, model: Model, audio_tokenizer: nn.Module, text_tokenizer: PreTrainedTokenizerFast):
    """Custom __init__ for the Generator class (from sesame csm repo)."""
    self._model = model
    self._model.setup_caches(1)

    self._text_tokenizer = text_tokenizer

    device = next(model.parameters()).device
    self._audio_tokenizer = audio_tokenizer.to(device=device)
    self.sample_rate = audio_tokenizer.sample_rate
    self.device = device

    self._watermarker = load_watermarker(device=device)


def reset_caches(model: Model):
    """Reset the caches of the model (used after each generation)."""
    model.reset_caches()
    for module in model.modules():
        if hasattr(module, "cache_enabled"):
            module.cache_enabled = False
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def lr_lambda(step: int) -> float:
    """Returns lambda for the learning rate scheduler for a linear warmup and decay."""
    if step < wandb.config["warmup_steps"]:
        return step / wandb.config["warmup_steps"]
    else:
        return (wandb.config["total_steps"] - step) / (wandb.config["total_steps"] - wandb.config["warmup_steps"])


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model.from_pretrained(args.model)
    model.forward = types.MethodType(forward, model)  # add the forward method to the model
    model = model.to(device=device, dtype=torch.bfloat16)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        notes=f"Config: {vars(args)}",
        config=vars(args),
        reinit=args.wandb_reinit,
        dir=args.output_dir / "wandb",
    )

    text_tokenizer = load_llama3_tokenizer()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(32)
    audio_tokenizer = mimi

    with open(args.data, "rb") as f:
        all_tokens = pickle.load(f)

    trainloader, valloader = create_dataloaders(all_tokens, wandb.config["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"])
    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    # Create a progress bar for all steps
    pbar = tqdm(total=wandb.config["total_steps"], desc="Training")
    epoch = 0
    train_losses = []
    model.train()
    for step, (tokens, tokens_mask) in enumerate(trainloader):
        epoch += tokens.size(0) / len(trainloader)
        tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)

        loss = model(tokens, tokens_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss = loss.item()
        train_losses.append(train_loss)

        if step % wandb.config["log_every"] == 0:
            wandb.log(
                {
                    "train_loss_avg": sum(train_losses) / len(train_losses),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )
            train_losses = []

        if step % wandb.config["val_every"] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_tokens, val_tokens_mask in valloader:
                    val_tokens, val_tokens_mask = val_tokens.to(device), val_tokens_mask.to(device)
                    val_loss += model(val_tokens, val_tokens_mask).item()

            val_loss /= len(valloader)
            wandb.log(
                {
                    "val_loss": val_loss,
                },
                step=step,
            )
            model.train()
            pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
        else:
            pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}
            )

        if step % wandb.config["gen_every"] == 0:
            model.eval()
            Generator.__init__ = types.MethodType(custom_generator_init, Generator)
            generator = Generator(model, audio_tokenizer, text_tokenizer)
            audio = generator.generate(
                text=wandb.config["gen_sentence"],
                speaker=args.gen_speaker,
                context=[],
                max_audio_length_ms=10_000,
            )

            wandb.log(
                {
                    "audio": wandb.Audio(audio.squeeze(0).cpu().numpy(), sample_rate=24_000),
                },
                step=step,
            )
            reset_caches(model)
            model.train()

        if step == wandb.config["total_steps"]:
            break

        pbar.update(1)

    # save the model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    pbar.close()
    wandb.finish()
