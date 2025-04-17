import argparse
import os
import pickle
from pathlib import Path
import yaml
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import plotly
import wandb
import torch
import torch.multiprocessing as mp

from finetune import finetune

def parse_args(arg_string=None):   
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/tokens.pkl", type=str, help="Path to the pre-tokenized data")
    parser.add_argument("--output_dir", type=Path, default="./sweep", help="Path to save the model")
    parser.add_argument("--model", type=str, default="sesame/csm-1b", help="Option to specify local path")
    parser.add_argument("--sweep_config", type=str, default="./configs/sweep.yaml", help="Path to the sweep config")
    parser.add_argument("--wandb_api_key", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="csm-sweep", help="Name of the project")
    parser.add_argument("--study_name", type=str, default="csm-sweep", help="Name of the study")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs to train before evaluating")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available)")
    parser.add_argument("--use_amp", type=bool, default=False, help="Use Automatic Mixed Precision Training")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--trials_per_gpu", type=int, default=1, help="Number of trials to run per GPU")
    parser.add_argument("--val_every", type=int, default=500, help="Number of steps between validation runs")
    parser.add_argument("--gen_every", type=int, default=0, help="Number of steps between generation runs")
    parser.add_argument("--save_every", type=int, default=0, help="Number of steps between saving the model")
    parser.add_argument("--log_every", type=int, default=10, help="Number of steps between logging the training loss")

    # Parameters for generation during evaluation
    parser.add_argument(
        "--gen_sentence",
        type=str,
        default="Bird law in this country is not governed by reason.",
        help="Sentence for model to generate during evaluation",
    )
    parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")

    args = parser.parse_args(arg_string.split() if arg_string else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def worker(args, gpu_id, study_name, storage_name, all_tokens):
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

    device = torch.device(f"cuda:{gpu_id}")
    
    def objective(trial):
        torch.cuda.set_device(gpu_id)
        
        if args.trials_per_gpu > 1:
            memory_fraction = 0.9 / args.trials_per_gpu
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)

        config = {}
        for name, param in sweep_config.items():
            if param["type"] == "categorical":
                config[name] = trial.suggest_categorical(name, param["values"])
            elif param["type"] == "float":
                config[name] = trial.suggest_float(name, float(param["min"]), float(param["max"]), log=param["log"])
            elif param["type"] == "int":
                config[name] = trial.suggest_int(name, int(param["min"]), int(param["max"]))
            elif param["type"] == "fixed":
                config[name] = eval(param["value"])
        
        wandb.init(
            project=args.wandb_project,
            name=f"trial-{trial.number}-gpu-{gpu_id}",
            config=config,
            group="optuna_sweep",
            dir=args.output_dir / "wandb",
            reinit=True,
        )

        best_val_loss = finetune(args, config, device, all_tokens, trial)
        wandb.finish()
        return best_val_loss
    
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
    parameter_pairs = [["learning_rate", "batch_size"], ["learning_rate", "decoder_loss_weight"], ["learning_rate", "weight_decay"], ["learning_rate", "max_grad_norm"]]
    for pair in parameter_pairs:
        try:
            contour_fig = plot_contour(study, params=pair)
            contour_fig.write_html(str(args.output_dir / f"{pair[0]}_{pair[1]}_contour.html"))
        except:
            print(f"Could not create contour plot for {pair[0]} and {pair[1]}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    mp.set_start_method('spawn')
    args = parse_args()
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.data, "rb") as f:
        all_tokens = pickle.load(f)
    
    storage_name = f"sqlite:///{args.output_dir}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name,
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
            args=(args, gpu_id, args.study_name, storage_name, all_tokens)
        )
        p.start()
        processes.append(p)
    
    # wait for all processes to finish
    for p in processes:
        p.join()

    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(study.best_trial.params, f, default_flow_style=False)
    
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