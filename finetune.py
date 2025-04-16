import argparse
import os
import pickle
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
from functools import partial
import optuna
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb

from utils.training import (
    load_model, 
    load_tokenizers, 
    generate_audio, 
    WarmupDecayLR,
    validate
)
from utils.data import create_dataloaders


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
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--gen_every", type=int, default=1000)
    parser.add_argument(
        "--gen_sentences",
        type=str,
        default="Bird law in this country is not governed by reason.",
        help="Sentence(s) for model to generate. If a path is provided, the model will generate from the sentences in the file.",
    )
    parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")

    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs to train. If not provided, the training will run indefinitely.")

    args = parser.parse_args(arg_string.split() if arg_string else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gen_sentences = Path(args.gen_sentences) if args.gen_sentences.endswith(".txt") else args.gen_sentences
    
    return args


def finetune(args: argparse.Namespace, config: dict, device: torch.device, all_tokens: dict, trial: optuna.Trial = None):
    """
    trial is only used when we are sweeping hyperparameters.
    """
    assert wandb.run is not None, "Wandb is not initialized"

    eff_batch_size = config["batch_size"] * config["grad_acc_steps"]
    
    # Load / create: model, tokenizers, dataloaders, optimizer, scheduler, and grad scaler.
    model = load_model(args.model, device)
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    trainloader, valloader = create_dataloaders(
        all_tokens, 
        config["batch_size"], 
        infinite_train=False,
    )
    total_steps = args.n_epochs * len(trainloader) if args.n_epochs else None
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = WarmupDecayLR(optimizer, config["warmup_steps"], total_steps, config["lr_decay"])
    scaler = GradScaler(enabled=args.use_amp)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "effective_batch_size": eff_batch_size,
        "config": config,
        "all_tokens": all_tokens,
        "args": args,
        "best_val_loss": float("inf"),
    }
    
    # Training loop
    step = 0
    train_losses = []
    pbar = tqdm(total=total_steps, desc="Training" if trial is None else f"Trial {trial.number}")
    model.train()
    
    for epoch in range(args.n_epochs):
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
                
            with autocast(device_type=str(device), dtype=torch.bfloat16, enabled=args.use_amp):
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
            
            if args.log_every and step % args.log_every == 0:
                wandb.log(
                    {
                        "train_loss_avg": sum(train_losses) / len(train_losses),
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
                train_losses = []

            if args.save_every and (step % args.save_every == 0 or step == total_steps - 1):
                state["model"] = model.state_dict()
                torch.save(state, args.output_dir / f"model_{step}.pt")
                if step == total_steps - 1:
                    torch.save(state, args.output_dir / f"model_final.pt")

            if args.val_every and (step % args.val_every == 0 or step == total_steps - 1):
                val_loss = validate(model, valloader, device, args.use_amp)
                wandb.log({"val_loss": val_loss}, step=step)

                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    torch.save(state, args.output_dir / f"model_bestval.pt")
                    wandb.save(args.output_dir / f"wandb_bestval.pt")

                # If this finetune is part of a sweep, report the validation loss to Optuna for pruning
                if trial is not None:
                    trial.report(val_loss, step)
                    if trial.should_prune():
                        wandb.finish()
                        pbar.close()
                        raise optuna.exceptions.TrialPruned()
                
                model.train()
                pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
            else:
                pbar.set_postfix(
                    {"train_loss": f"{train_loss:.4f}", "learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}
                )
            
            if args.gen_every and step % args.gen_every == 0:
                gen_sentences = []
                if isinstance(args.gen_sentences, str):
                    gen_sentences.append(args.gen_sentences)
                elif isinstance(args.gen_sentences, Path):
                    with open(args.gen_sentences, "r") as f:
                        gen_sentences = f.readlines()

                for i, sentence in enumerate(gen_sentences):
                    audio, wer = generate_audio(
                        model,
                        audio_tokenizer,
                        text_tokenizer,
                        sentence,
                        args.gen_speaker,
                        device,
                        use_amp=args.use_amp
                    )
                    
                    wandb.log(
                        {
                            f"audio_{i}": wandb.Audio(audio, sample_rate=24_000),
                            f"wer_{i}": wer,
                        },
                        step=step,
                    )
                model.train()
            
            pbar.update(1)
            
            if step >= total_steps:
                break
            
            step += 1
    
    pbar.close()
    return val_loss


if __name__ == "__main__":
    args = parse_args()
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.data, "rb") as f:
        all_tokens = pickle.load(f)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"training_bs-{config['batch_size']}x{config['grad_acc_steps']}",
        notes=f"Config: {config}",
        config={**config, **vars(args)},
        reinit=args.wandb_reinit,
        dir=args.output_dir / "wandb",
    )

    final_val_loss = finetune(args, config, device, all_tokens)

    wandb.finish()
