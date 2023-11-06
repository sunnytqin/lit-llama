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
    precomputed_large_emb_dir_val: str,
    output_dir: str,
    precomputed_head_input_emb_dir_val: str = None,
    model_size: str = "7B",
    small_checkpoint_path: str = None,
    large_checkpoint_path: str = None,
    hidden_dim: int = 2048,
    no_hidden_layers: int = 5,
    dropout: float = 0.1,
    activation: str = "relu",
    glue_lm_head: bool = False,
    no_bins: int = 2,
    min_bin: float = 0,
    max_bin: float = 1, # JSD is bounded by ln(2)
    min_entropy: float = None,
    max_entropy: float = None,
    provide_entropy_as_input: bool = False,
    target_fn_name: str = "large_entropy",
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
    if not large_checkpoint_path:
        large_checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth')
    else:
        large_checkpoint_path = Path(large_checkpoint_path)

    assert small_checkpoint_path.is_file()
    assert large_checkpoint_path.is_file()

    # Load the (small) LM heads of both the small and the large model.
    # We've only cached the embeddings and not the (much larger) logits.
    small_lm_head = load_lm_head(small_checkpoint_path, dtype=DTYPE, device=DEVICE)
    large_lm_head = load_lm_head(large_checkpoint_path, dtype=DTYPE, device=DEVICE)

    # Load the precomputed logits
    shard_dirs = [
        precomputed_small_emb_dir_val,
        precomputed_large_emb_dir_val,
    ]

    if(precomputed_head_input_emb_dir_val):
        shard_dirs.append(precomputed_head_input_emb_dir_val)

    logit_loader = PrecomputedShardLoader(shard_dirs, dataset_filter_path=dataset_filter_path)

    # Initialize the model
    shared_head_params = {
        "no_bins": no_bins,
        "hidden_dim": hidden_dim,
        "no_hidden_layers": no_hidden_layers,
        "dropout": dropout,
        "activation": activation,
    }
    if(not glue_lm_head):
        distance_prediction_head = DistancePredictionHead(
            input_dim=small_lm_head.weight.shape[1],
            **shared_head_params,
        )
    else:
        distance_prediction_head = DistancePredictionHeadWithLMHead(
            lm_head=small_lm_head,
            **shared_head_params,
        )

    distance_prediction_head.to(DEVICE)

    # Umm uhh
    distance_prediction_head = torch.compile(distance_prediction_head)

    checkpoint = torch.load(head_checkpoint_path)
    distance_prediction_head.load_state_dict(checkpoint)

    param_count = sum(
        p.numel() for p in distance_prediction_head.parameters() if p.requires_grad
    )
    logging.info(f"Loaded prediction head ({param_count} parameters)...")

    # Where we smuggle out data from the preprocessor
    stash = {}

    val_data_gen = _preprocessor(
        shard_loader=logit_loader,
        small_lm_head=small_lm_head,
        large_lm_head=large_lm_head,
        no_bins=no_bins,
        min_bin=min_bin,
        max_bin=max_bin,
        min_entropy=min_entropy,
        max_entropy=max_entropy,
        provide_entropy_as_input=provide_entropy_as_input,
        target_fn_name=target_fn_name,
        device=DEVICE,
        dtype=DTYPE,
        _stash=stash,
    )

    val_bl = batch_loader(
        data_gen=val_data_gen,
        batch_size=128,
        skip_frac=0.,
    )

    with torch.no_grad():
        val_acc_sum = 0
        val_batch_count = 0
        all_val_preds = []
        all_val_gt = []
        for j, (val_inputs, val_targets) in enumerate(val_bl):
            val_inputs = val_inputs.to(DEVICE)
            val_targets = val_targets.to(DEVICE)

            val_inputs = val_inputs.to(torch.float32)
            val_targets = val_targets.to(torch.int64)

            val_outputs = distance_prediction_head(val_inputs)

            val_preds = torch.argmax(val_outputs, dim=-1)

            val_acc = val_preds == val_targets
            val_acc_sum += torch.mean(val_acc.float()).item()

            val_batch_count += 1

            all_val_preds.extend(val_preds.cpu().tolist())
            all_val_gt.extend(val_targets.cpu().tolist())

    # The last batch is dropped, so we trim
    for k in stash:
        stash[k] = stash[k][:len(all_val_preds)]

    assert(len(list(stash.values())[0]) == len(all_val_preds))
    stash["val_preds"] = all_val_preds
    stash["val_gt"] = all_val_gt

    keys = list(stash.keys())
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "val_preds.txt"), "w") as fp:
        joined_keys = '\t'.join(keys)
        fp.write(f"{joined_keys}\n")
        for t in zip(*[stash[k] for k in keys]):
            joined = '\t'.join([str(x) for x in t])
            fp.write(f"{joined}\n")


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
