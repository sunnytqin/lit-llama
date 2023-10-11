import copy
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
import warnings

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import wandb

from lit_llama import LLaMA, Tokenizer
from train_head_utils import (
    batch_loader,
    DistancePredictionHead,
    DistancePredictionHeadWithLMHead,
    load_lm_head,
    PrecomputedShardLoader,
    _preprocessor,
)


DTYPE = torch.float32
DEVICE = torch.device("cuda:0")


def main(
    *,
    head_checkpoint_path: str,
    precomputed_small_emb_dir_val: str,
    output_dir: str,
    model_size: str = "7B",
    small_checkpoint_path: str = None,
    hidden_dim: int = 2048,
    no_hidden_layers: int = 1,
    dropout: float = 0.1,
    activation: str = "relu",
    no_bins: int = 2,
    min_bin: float = -7,
    max_bin: float = np.log(np.log(2)), # JSD is bounded by ln(2)
    min_entropy: float = None,
    max_entropy: float = None,
    provide_entropy_as_input: bool = False,
    target_fn_name: str = "log_jsd",
    dataset_filter_path: str = None,
    seed: int = 42,
) -> None:
    # MUST COME FIRST
    args = locals()

    torch.manual_seed(seed)
    random.seed(seed)

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)

    if not small_checkpoint_path:
        small_checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth')
    else:
        small_checkpoint_path = Path(small_checkpoint_path)

    assert small_checkpoint_path.is_file()

    # Load the (small) LM heads of both the small and the large model.
    # We've only cached the embeddings and not the (much larger) logits.
    small_lm_head = load_lm_head(small_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type="llama", model_size=model_size)

    # Load the precomputed logits
    shard_dirs = [
        precomputed_small_emb_dir_val,
    ]

    logit_loader = PrecomputedShardLoader(shard_dirs, dataset_filter_path=dataset_filter_path)

    # Initialize the model
    shared_head_params = {
        "no_bins": no_bins,
        "hidden_dim": hidden_dim,
        "no_hidden_layers": no_hidden_layers,
        "dropout": dropout,
        "activation": activation,
    }

    distance_prediction_head = DistancePredictionHead(
        input_dim=small_lm_head.weight.shape[1],
        **shared_head_params,
    )

    distance_prediction_head.to(DEVICE)
    distance_prediction_head.eval()

    print(distance_prediction_head)

    # Umm uhh
    distance_prediction_head = torch.compile(distance_prediction_head)

    checkpoint = torch.load(head_checkpoint_path)
    # print(head_checkpoint_path)
    # print(checkpoint)
    # print(checkpoint.keys())
    # print(distance_prediction_head.state_dict().keys())
    distance_prediction_head.load_state_dict(checkpoint)

    param_count = sum(
        p.numel() for p in distance_prediction_head.parameters() if p.requires_grad
    )
    logging.info(f"Loaded prediction head ({param_count} parameters)...")

    # Where we smuggle out data from the preprocessor
    stash = {}

    pred_count = {}
    entropy_sums = {}
    for i, shard_tups in enumerate(logit_loader):
        small_tup = shard_tups[0]
        small_key, small_emb = small_tup
        assert(len(small_emb.shape) == 3)
        small_emb = small_emb[:, -1] # Take the last token
        small_emb = small_emb.to(device=DEVICE, dtype=DTYPE)
        small_logits = torch.nn.functional.softmax(small_lm_head(small_emb), dim=-1)
        small_entropy = torch.sum((-small_logits) * torch.log(small_logits), dim=-1)

        val_outputs = distance_prediction_head(small_emb)
        val_preds = torch.argmax(val_outputs, dim=-1)
        for p in val_preds:
            p = str(p.item())
            pred_count[p] = pred_count.get(p, 0) + 1
            entropy_sums[p] = entropy_sums.get(p, 0) + small_entropy.item()

    print(pred_count)
    for p, c in pred_count.items():
        print(p, entropy_sums[p] / c)



if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    warnings.filterwarnings(
        # SLURM srun warning
        "ignore", 
        message="The `srun` command is available on your system but is not used",
    )

    CLI(main)
