import json
import pathlib
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

directory = pathlib.Path(__file__)
sys.path.append(str(directory.parent.parent))
from train_head_utils import (
    load_lm_head,
    PrecomputedShardLoader,
)

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENTROPY_BAND_RECTANGLE = True

llama_checkpoint = lambda s: f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{s}/lit-llama.pth"
llama_logit_path = lambda s: f"data/wikipedia_scratch/wiki_logits/{s}_val"
llama_2_checkpoint = lambda s: f"/n/holyscratch01/barak_lab/Everyone/lit-gpt_llama_2/Llama-2-{s}-hf/"
llama_2_checkpoint_dummy = lambda s: f"/n/holyscratch01/barak_lab/Everyone/lit-gpt_llama_2/Llama-2-{s}-hf-dummy"
llama_2_logit_path = lambda s: f"data/wikipedia_scratch/wiki_logits/llama_2_{s}_val"
pythia_checkpoint = lambda s: f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-{s}/"
pythia_logit_path = lambda s: f"data/wikipedia_scratch/wiki_logits/pythia-{s}_val"

inputs = [
    {
        "inputs": {
            "model_type": "llama",
            "x_model_size": "7B",
            "x_checkpoint_path": llama_checkpoint("7B"),
            "x_logit_path": llama_logit_path("7B"),
            "y_model_size": "65B",
            "y_checkpoint_path": llama_checkpoint("65B"),
            "y_logit_path": llama_logit_path("65B"),
        },
        "label_x": "LLaMA 7B",
        "label_y": "LLaMA 65B",
        "title": "Wikipedia entropy"
    },
    {
        "inputs": {
            "model_type": "llama_2",
            "x_model_size": "7b",
            "x_checkpoint_path": llama_2_checkpoint("7b"),
            "x_logit_path": llama_2_logit_path("7b"),
            "y_model_size": "70b",
            "y_checkpoint_path": llama_2_checkpoint_dummy("70b"),
            "y_logit_path": llama_2_logit_path("70b"),
        },
        "label_x": "LLaMA 2 7B",
        "label_y": "LLaMA 2 70B",
        "title": "Wikipedia entropy"
    },
    {
        "inputs": {
            "model_type": "pythia",
            "x_model_size": "1.4b",
            "x_checkpoint_path": pythia_checkpoint("1.4b"),
            "x_logit_path": pythia_logit_path("1.4b"),
            "y_model_size": "12b",
            "y_checkpoint_path": pythia_checkpoint("12b"),
            "y_logit_path": pythia_logit_path("12b"),
        },
        "label_x": "Pythia 1.4B",
        "label_y": "Pythia 12B",
        "title": "Pile entropy"
    }
]


def get_entropies(
    model_type, 
    x_model_size, 
    x_checkpoint_path, 
    x_logit_path, 
    y_model_size, 
    y_checkpoint_path, 
    y_logit_path
):
    x_lm_head = load_lm_head(x_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=x_model_size)
    y_lm_head = load_lm_head(y_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=y_model_size)

    loader = PrecomputedShardLoader(
        [x_logit_path, y_logit_path],
    )

    x_entropies = []
    y_entropies = []
    for tups in loader:
        x_tup, y_tup = tups

        x_key, x_emb = x_tup
        y_key, y_emb = y_tup

        assert(x_key == y_key)

        x_emb = x_emb.to(device=DEVICE, dtype=DTYPE)
        y_emb = y_emb.to(device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            # Compute logits from the small model embeddings
            x_logits = x_lm_head(x_emb)
            y_logits = y_lm_head(y_emb)

            # Softmax both sets of logits
            x_logits_softmax = torch.nn.functional.softmax(x_logits, dim=-1)
            y_logits_softmax = torch.nn.functional.softmax(y_logits, dim=-1)

            x_logs = torch.nn.functional.log_softmax(x_logits, dim=-1)
            x_entropy = torch.sum(-1 * x_logits_softmax * x_logs, dim=-1)
            y_logs = torch.nn.functional.log_softmax(y_logits, dim=-1)
            y_entropy = torch.sum(-1 * y_logits_softmax * y_logs, dim=-1)

            x_entropies.extend(x_entropy.cpu().float().unbind())
            y_entropies.extend(y_entropy.cpu().float().unbind())

    return x_entropies, y_entropies

# entropies = []
# for inp in inputs:
#     entropies.append(get_entropies(**inp["inputs"]))

# # Pickle checkpoint
# with open("pickles/entropy_scatter.pickle", "wb") as fp:
#     pickle.dump(entropies, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open("pickles/entropy_scatter.pickle", "rb") as fp:
    entropies = pickle.load(fp)

fig, axs = plt.subplots(1, len(inputs), figsize=(3 * len(inputs), 3))
if(len(inputs) == 1):
    axs = [axs]

for i, inp in enumerate(inputs):
    x_entropies, y_entropies = entropies[i]
    xmax, ymax = 10, 10
    hb = axs[i].hexbin(x_entropies, y_entropies, gridsize=50, bins="log", cmap="inferno")
    axs[i].set_xlim([0, xmax])
    axs[i].set_ylim([0, ymax])
    axs[i].set_xlabel(inp["label_x"])
    axs[i].set_ylabel(inp["label_y"])
    axs[i].set_title(inp["title"])
    axs[i].margins(x=0, y=0)

    if(i != 0):
        axs[i].set_yticks([])

if(ENTROPY_BAND_RECTANGLE):
    axs[-1].axvspan(2.5, 3.0, color="green", alpha=0.4, lw=0)

fig.savefig("figures/entropy_scatter.pdf", bbox_inches="tight")

