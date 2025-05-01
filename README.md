<h1 align="center">
    <img src="media/speechmatics_logo.png" alt="Speechmatics" width="600">
</h1>

![Cover](media/cover.png)

# Training and finetuning of Sesame AI's Conversational Speech Model.

Use this repository to finetune Sesame's CSM-1B into new languages or voices, or train it from scratch. [Read blog post here](https://blog.speechmatics.com/sesame-finetune).

**Features**:
- Efficient training via: pre-tokenization, compute amortization, padding-minimized batching. Supports both partial and full loading of data (only use partial loading for large datasets that cannot fit in memory).
- Finetune by modifying the original weights (as opposed to LoRA). This has a higher compute burden but is much better for significant domain shifts like new languages.
- Hyperparameter optimization using Optuna. 
- Performance enhancements: Gradient clipping, accumulation, mixed precision training, advanced LR scheduling, experiment tracking.

## Installation

1. Clone this repo and set up a virtual environment:

```bash
git clone https://github.com/knottwill/sesame-finetune.git
cd sesame-finetune
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Clone the [official CSM repo](https://github.com/SesameAILabs/csm)

```bash
git clone https://github.com/SesameAILabs/csm.git ~/csm
cd ~/csm
git checkout 836f886515f0dec02c22ed2316cc78904bdc0f36
cd -
```

3. Create environment file 

Copy the example (must be in the root of the repo):

```bash
cp .env.example .env
```

Fill it in with a [Weights&Biases API key](https://docs.wandb.ai/support/find_api_key/) and the path to the CSM repo.

```dotenv
WANDB_API_KEY=your-key-here
CSM_REPO_PATH=~/csm
```

## Usage

**Data preparation and pre-tokenization**

Prepare your dataset with train set and validation set metadata files with each entry in the files containing: the path to an audio file (must be `.wav`), the text transcription, start / end times of the transcription in the wav file (optional), and the speaker ID (optional). Several formats for this metadata file are supported (`.json`, `.csv`, `.sql`, `.parquet`, `.hdf5`). An example `metadata.json` file might look like:

```json
  {
    "text": "They had obvious value as wonders of the natural world.",
    "path": "/data/utterance_0.wav",
  },
  {
    "text": "and ten years later the Fort Worth and Rio Grande Railroad laid tracks in the county.",
    "path": "/data/long_audio.wav",
    "start": 171.1,
    "end": 182.6,
    "speaker": 30,
  },
```

Since we will want to train for several epochs, it is more efficient to pre-tokenize all the data before starting the training run:

```bash
python pretokenize.py --train_data /path/to/train/metadata.json --val_data /path/to/val/metadata.json --output /path/to/tokenized/data.hdf5
```

**Train / Finetune**

To finetune the model, you will need to provide the pre-tokenized data, a finetuning hyperparameters config file, a Weights & Biases API key to track the experiment, the number of epochs to train for, and what sentence to use for generations. The script will generate every `--gen_every` steps, and log the resulting audio to Weights & Biases. 

```bash
python train.py --data /path/to/tokenized/data.hdf5 --config ./configs/finetune_param_defaults.yaml --n_epochs 25 --gen_every 500 --gen_sentence "Marie aime les pommes et les poires."
```

If you want to train from randomly initialized weights, use `--train_from_scratch`. 

If your dataset is too large to fit in memory, use `--partial_data_loading`.

**(Optional) Hyperparameter sweep**

To sweep finetuning hyperparameters, specify the path to the pre-tokenized data, an experiment directory, the number of epochs to run for each trial, the number of trials, and the number of GPUs (for parallelism of trials). 

```bash
python sweep.py --data /path/to/tokenized/data.hdf5 --sweep_config ./configs/sweep.yaml --output_dir ./my-sweep --n_epochs 3 --n_trials 50 --n_gpus 2
```

Weights & Biases is used for comparing the sweeps, like so:

![sweeps](media/sweep_tracking.png)