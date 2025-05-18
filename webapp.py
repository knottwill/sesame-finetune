"""
Create a simple web app such that prior to loading I specify:  
- Path to model checkpoint  
- Path to prompt wav file, with start and end point, as well as the transcription of that prompt.  
  
When I load the web app it should do some warmup generations.  
  
Then the user can enter text and have audio generate (ideally with a bar tracking generation progress).
"""

from src.generator import load_csm_1b
from pathlib import Path
import pandas as pd
import sqlite3
import torch
import torchaudio
from src.generator import Segment
from src.data import load_metadata, load_audio
from IPython.display import Audio

# input from user
text = "Kannst dich gerne uns anschließen."
language = "German" # or "French" or "Swedish"

datasets = {
    "German": {
        "model": "/exp/willk/tts/csm/ft_de_seraphina_10h/model_5000.pt",
        "train": "/data/artefacts/tts/edge_tts/de_DE_Seraphina/data_train_10h/train.sql",
        "val": "/data/artefacts/tts/edge_tts/de_DE_Seraphina/data_train_10h/val.sql",
        "prompt_index": 2
    },
    "Swedish": {
        "model": "/exp/willk/tts/csm/ft_sv_mattias_10h_L40/model_5000.pt",
        "train": "/data/artefacts/tts/edge_tts/sv_SE_Mattias/data_train_10h/train.sql",
        "val": "/data/artefacts/tts/edge_tts/sv_SE_Mattias/data_train_10h/val.sql"
    },
    "French": {
        "model": "/exp/willk/tts/csm/9_may_fr_am_200hr_finetune/model_10000.pt",
        "train": "/data/artefacts/tts/fr/2025-03-20_2000hrs_am_data/data_train_words/train.sql",
        "val": "/data/artefacts/tts/fr/2025-03-20_2000hrs_am_data/data_train_words/val.sql",
        "prompt_index": 2
    }
}

### Load model and warm up 
dataset = datasets[language]["model"]
generator = load_csm_1b(datasets[language]["model"])

val_metadata = load_metadata(datasets[language]["val"])
val_metadata = val_metadata[val_metadata['duration'].between(2, 5)]
prompt = val_metadata.iloc[datasets[language]["prompt_index"]]

context = [
    Segment(
        text=prompt['text'],
        speaker=999,
        audio=load_audio(prompt['path'], prompt['start'], prompt['end'])
    )
]

warmup_text = {
    "German": "Solange du hier bist, denke ich, dass es funktionieren wird.",
    "French": "Je suis heureux de te revoir.",
    "Swedish": "Jag är glad att du är här."
}[language]
out = generator.generate(text=warmup_text, speaker=999, context=context)
# Audio(out.squeeze().cpu().numpy(), rate=24000)

### Actually generate audio if the user has entered text and clicked the button
audio = generator.generate(text=text, speaker=999, context=context)
audio = audio.squeeze().cpu().numpy()

### Show audio on the web page
