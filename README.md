# Finetune Sesame AI's Conversational Speech Model.

### Installation

Clone the repo and set up a virtual environment:
```bash
git clone https://github.com/knottwill/sesame-finetune.git
cd sesame-finetune
python3.10 -m venv .venv
source .venv/bin/activate
```

Install the dependencies with:
```bash
./install_dependencies.sh
```
This clones the official sesame CSM repo, checks out the right commit, installs their requirements, installs the repo as a package, then installs our additional requirements.

### Usage

**Data and pre-tokenization**

You will need a dataset to finetune on. In the code I am making the assumption that the dataset will contain a metadata file with each entry giving us the path to audio wav file, text transcription, and optionally the start / end times of the transcription in the wav file. There can also optionally be a speaker ID. Many formats for this metadata file are possible (`.json`, `.csv`, `.sql`, `.parquet`, `.pkl`). An example `metadata.json` file might look like:

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

When we have our train and validation set metadata files, we first pre-tokenize all the data:

```bash
python pretokenize.py --train_data /path/to/train/metadata.json --val_data /path/to/val/metadata.json --output /path/to/output/tokens.pkl
```

**(Optional) Hyperparameter sweep**

```bash
python sweep.py 
```

**Finetune**

```bash
python finetune.py
```
