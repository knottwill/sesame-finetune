import argparse
import os
import sys
import pickle
from pathlib import Path
import yaml
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import plotly
import wandb
import torch
from torch.amp import autocast
from tqdm import tqdm
import torch.multiprocessing as mp

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
    parser.add_argument("--model", type=str, default="sesame/csm-1b", help="Option to specify local path")
    parser.add_argument("--wandb_api_key", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="csm-sweep", help="Name of the project")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs to train before evaluating")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available)")
    parser.add_argument("--use_amp", type=bool, default=True, help="Use Automatic Mixed Precision Training")
    parser.add_argument("--n_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--trials_per_gpu", type=int, default=1, help="Number of trials to run per GPU")
    parser.add_argument("--val_every", type=int, default=100, help="Number of steps between validation runs")

    # Parameters for generation during evaluation
    parser.add_argument(
        "--gen_sentence",
        type=str,
        default="Bird law in this country is not governed by reason.",
        help="Sentence for model to generate during evaluation",
    )
    parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")

    return parser.parse_args(arg_string.split() if arg_string else None)


args = parse_args()


# The train_and_evaluate function needs to be modified to accept GPU ID
def train_and_evaluate(trial, all_tokens, gpu_id):
    """
    Train the model with trial-specific hyperparameters and return validation loss.
    """
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    
    # Limit GPU memory usage to allow multiple trials per GPU
    if args.trials_per_gpu > 1:
        memory_fraction = 0.9 / args.trials_per_gpu
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    # Configure hyperparameters for this trial
    config = {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 5.0),
        "grad_acc_steps": 1,
        "use_amp": args.use_amp,
        "gen_sentence": args.gen_sentence,
    }
    
    # Create a unique run name for this trial including GPU ID
    run_name = f"trial-{trial.number}-gpu-{gpu_id}"
    
    # Initialize wandb for this trial
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=config,
        group="optuna_sweep",
        dir=args.output_dir / "wandb",
        reinit=True,
    )
    
    device = torch.device(f"cuda:{gpu_id}")
    model = load_model(args.model, device)
    
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    
    trainloader, valloader, _ = create_dataloaders(all_tokens, config["batch_size"], infinite_train=False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    total_steps = len(trainloader) * args.n_epochs
    config["total_steps"] = total_steps
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda step: lr_lambda(step, config["warmup_steps"], total_steps)
    )
    
    model.train()
    pbar = tqdm(total=total_steps, desc=f"Trial {trial.number} GPU {gpu_id}")
    step = 0
    
    for epoch in range(args.n_epochs):
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
            
            with autocast(device_type=str(device), enabled=args.use_amp):
                loss = model(tokens, tokens_mask)
                loss = loss / config["grad_acc_steps"]
            
            loss.backward()
            
            if (step + 1) % config["grad_acc_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss = loss.item()
            wandb.log({"train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}, step=step)
            pbar.update(1)
            pbar.set_postfix({"loss": f"{train_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
            
            step += 1
            
            # Run validation and pruning every val_every steps
            if step % args.val_every == 0:
                val_loss = validate(model, valloader, device, args.use_amp)
                wandb.log({"val_loss": val_loss}, step=step)
                
                # Report to Optuna for pruning
                trial.report(val_loss, step)
                if trial.should_prune():
                    wandb.finish()
                    pbar.close()
                    raise optuna.exceptions.TrialPruned()
                    
                model.train()
    
    # Final evaluation  
    val_loss = validate(model, valloader, device, args.use_amp)
    wandb.log({"final_val_loss": val_loss})
    
    try:
        audio = generate_audio(
            model,
            audio_tokenizer,
            text_tokenizer,
            config["gen_sentence"],
            args.gen_speaker,
            device, 
            use_amp=args.use_amp,
            max_audio_length_ms=5_000,  # Shorter for sweep
        )
        
        wandb.log({"audio": wandb.Audio(audio.squeeze(0).cpu().numpy(), sample_rate=24_000)})
    except Exception as e:
        print(f"Audio generation failed: {e}")
    
    wandb.finish()
    pbar.close()
    
    model_path = args.output_dir / f"model_trial_{trial.number}_gpu_{gpu_id}.pt"
    torch.save(model.state_dict(), model_path)

    return val_loss


def worker(gpu_id, study_name, storage_name, all_tokens):
    """
    Worker function for each GPU to run multiple trials.
    """
    torch.cuda.set_device(gpu_id)
    
    trials_per_worker = args.n_trials // (args.n_gpus * args.trials_per_gpu)
    if trials_per_worker == 0:
        trials_per_worker = 1
    
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )
    
    def objective(trial):
        return train_and_evaluate(trial, all_tokens, gpu_id)
    
    study.optimize(objective, n_trials=trials_per_worker)


def save_visualization(study):
    """
    Save Optuna visualization plots to the output directory.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimization history plot
    history_fig = plot_optimization_history(study)
    history_fig.write_html(str(args.output_dir / "optimization_history.html"))
    
    # Parameter importance plot
    param_importance_fig = plot_param_importances(study)
    param_importance_fig.write_html(str(args.output_dir / "param_importance.html"))
    
    # Contour plots for pairs of parameters
    try:
        contour_fig = plot_contour(study, params=["learning_rate", "batch_size"])
        contour_fig.write_html(str(args.output_dir / "contour_plot.html"))
    except:
        print("Could not create contour plot")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    mp.set_start_method('spawn')
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.data, "rb") as f:
        all_tokens = pickle.load(f)
    
    study_name = f"csm-sweep-{args.n_epochs}epochs"
    storage_name = f"sqlite:///{args.output_dir}/optuna.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    processes = []
    total_workers = args.n_gpus * args.trials_per_gpu
    
    for i in range(total_workers):
        gpu_id = i % args.n_gpus
        p = mp.Process(
            target=worker,
            args=(gpu_id, study_name, storage_name, all_tokens)
        )
        p.start()
        processes.append(p)
    
    # wait for all processes to finish
    for p in processes:
        p.join()

    config = {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value
    }
    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    save_visualization(study)
    
    # log a summary of the sweep to wandb
    wandb.init(
        project=args.wandb_project,
        name="sweep-summary",
        dir=args.output_dir / "wandb",
    )
    wandb.config.update(study.best_trial.params)
    wandb.log({"best_val_loss": study.best_trial.value})
    
    param_importance_fig = plot_param_importances(study)
    wandb.log({"param_importance": wandb.Html(plotly.io.to_html(param_importance_fig))})
    
    history_fig = plot_optimization_history(study)
    wandb.log({"optimization_history": wandb.Html(plotly.io.to_html(history_fig))})
    
    wandb.finish()