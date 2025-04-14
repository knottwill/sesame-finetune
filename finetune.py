import argparse
import os
import pickle
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb

from utils import (
    load_model, 
    load_tokenizers, 
    generate_audio, 
    lr_lambda,
    validate
)
from dataloaders import create_dataloaders


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/tokens.pkl", type=str, help="Path to the pre-tokenized data")
    parser.add_argument("--output_dir", type=Path, default="./exp", help="Path to save the model")
    parser.add_argument("--config", type=str, default='./configs/default.yaml', help="Path to the finetuning config")
    parser.add_argument(
        "--model",
        type=str,
        default="sesame/csm-1b",
        help="Pretrained model name or local path",
    )

    parser.add_argument("--wandb_api_key", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="csm-finetuning", help="Name of the project")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name of the run")
    parser.add_argument("--wandb_reinit", type=bool, default=True, help="Whether to reinitialize the run")

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
    
    # Training arguments
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--lr_decay", type=str, default="linear", choices=["linear", "constant", "exponential", "cosine"],
                        help="Learning rate decay type after warmup")
    
    return parser.parse_args(arg_string.split() if arg_string else None)


if __name__ == "__main__":
    args = parse_args()
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add learning rate decay type
    config["lr_decay"] = args.lr_decay
    
    # Log effective batch size
    effective_batch_size = config["batch_size"] * config["grad_acc_steps"]
    print(f"Effective batch size: {effective_batch_size}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"training_bs-{config['batch_size']}x{config['grad_acc_steps']}",
        notes=f"Config: {config}",
        config={**config, **vars(args)},
        reinit=args.wandb_reinit,
        dir=args.output_dir / "wandb",
    )
    
    # Make output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizers
    model = load_model(args.model, device)
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    
    # Load data
    with open(args.data, "rb") as f:
        all_tokens = pickle.load(f)
    
    # Setup dataloaders
    trainloader, valloader = create_dataloaders(
        all_tokens, 
        config["batch_size"], 
        infinite_train=True,
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Set total steps for the scheduler
    total_steps = config["total_steps"]
    
    lr_lambda_fn = partial(
        lr_lambda, 
        warmup_steps=config["warmup_steps"], 
        total_steps=total_steps,
        decay_type=config["lr_decay"]
    )
    scheduler = LambdaLR(optimizer, lr_lambda_fn, last_epoch=-1)
    scaler = GradScaler(enabled=config["use_amp"])
    
    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training")
    
    # Training loop
    epoch = 0
    train_losses = []
    model.train()
    
    for step, (tokens, tokens_mask) in enumerate(trainloader):
        tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
        
        with autocast(device_type=str(device), enabled=config["use_amp"]):
            # Forward pass
            loss = model(tokens, tokens_mask)
            loss = loss / config["grad_acc_steps"]
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config["grad_acc_steps"] == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        if step % config["log_every"] == 0:
            wandb.log(
                {
                    "train_loss_avg": sum(train_losses) / len(train_losses),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )
            train_losses = []
        
        if step % config["val_every"] == 0:
            val_loss = validate(model, valloader, device, config["use_amp"])
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
        
        if step % config["gen_every"] == 0:
            # Generate audio
            audio = generate_audio(
                model,
                audio_tokenizer,
                text_tokenizer,
                args.gen_sentence,
                args.gen_speaker,
                device,
                use_amp=config["use_amp"]
            )
            
            wandb.log(
                {
                    "audio": wandb.Audio(audio.squeeze(0).cpu().numpy(), sample_rate=24_000),
                },
                step=step,
            )
            model.train()
        
        pbar.update(1)
        
        if step >= total_steps:
            break
        
        # Update epoch counter
        epoch = step // len(trainloader) + 1
    
    # Save the model
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    
    pbar.close()
    wandb.finish()
