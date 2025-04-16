<h1 align="center">
    <img src="media/speechmatics_logo.png" alt="Speechmatics" width="600">
</h1>

# Finetune Sesame AI's Conversational Speech Model.

### Installation

Clone the repo and set up a virtual environment:
```bash
git clone https://github.com/knottwill/sesame-finetune.git
cd sesame-finetune
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Annoyingly, the [CSM repo](https://github.com/SesameAILabs/csm) is not set up to be installable as a package (despite the `setup.py`), to use the code we need to manually clone the repo and add it to our `sys.path`. 

```bash
git clone https://github.com/SesameAILabs/csm.git ~/csm
export CSM_PATH='~/csm'
echo "export CSM_PATH='~/csm'" >> .venv/bin/activate
```

Now when we want to import code from the CSM repo we can use `sys.path.append(os.getenv("CSM_PATH", "~/csm"))`. 

### Usage

**Data preparation and pre-tokenization**

Prepare your dataset with train set and validation set metadata files with each entry in the files containing: the path to an audio file (must be `.wav`), the text transcription, start / end times of the transcription in the wav file (optional), and the speaker ID (optional). Several formats for this metadata file are supported (`.json`, `.csv`, `.sql`, `.parquet`, `.pkl`). An example `metadata.json` file might look like:

```json
  {
    "text": "They had obvious value as wonders of the natural world.",
    "path": "/data/utterance_0.wav",
  },
  {
    "text": "At the time, Niagara Falls was a thriving community.",
    "path": "/data/utterance_1.wav",
  },
  {
    "text": "and ten years later the Fort Worth and Rio Grande Railroad laid tracks in the county.",
    "path": "/data/long_audio.wav",
    "start": 171.1,  # Start point (optional)
    "end": 182.6,    # End point (optional)
    "speaker": 30,   # Speaker id (optional)
  },
```

Since we will want to train for several epochs, it is more efficient to pre-tokenize all the data before starting the training run:

```bash
python pretokenize.py --train_data /path/to/train/metadata.json --val_data /path/to/val/metadata.json --output /path/to/tokenized/data.pkl
```

**(Optional) Hyperparameter sweep**

To perform a hyperparameter sweep, specify the path to the pre-tokenized data, an experiment directory, the number of epochs to run for each trial, the number of trials, and the number of GPUs (for parallelism of trials). You will also need to provide a Weights & Biases API key for comparing the sweeps. 

```bash
python sweep.py --data /path/to/tokenized/data.pkl --sweep_config ./configs/sweep.yaml --output_dir ./my-sweep --n_epochs 3 --n_trials 50 --n_gpus 2 --wandb_api_key WANDB_API_KEY
```

**Finetune**

To finetune the model, you will need to provide the pre-tokenized data, a finetuning hyperparameters config file, a Weights & Biases API key to track the experiment, the number of epochs to train for, and what sentence to use when generating.

```bash
python finetune.py --data /path/to/tokenized/data.pkl --config ./configs/default.yaml --n_epochs 25 --gen_every 500 --gen_sentence "Marie aime les pommes et les poires." --wandb_api_key WANDB_API_KEY
```
