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
# import wandb

from lit_llama import LLaMA, Tokenizer
from train_head_utils import (
    batch_loader,
    load_lm_head,
    PrecomputedShardLoader,
    _preprocessor,
)

"""
This script creates a filter for a joint embedding dataset of the kind produced by precompute_logits.py.
Specifically, it identifies a subset of the data such that the small model entropy is in a given range
and then subsamples from this subset until the proportion of examples with low large model entropy is
approximately equal to the proportion of examples with high large model entropy.
"""


# DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
DTYPE = torch.float32
DEVICE = torch.device("cuda:0")


def main(
    *,
    precomputed_small_emb_dir: str,
    precomputed_large_emb_dir: str,
    output_dir: str,
    small_checkpoint_path: str = None,
    large_checkpoint_path: str = None,
    entropy_min: float = 2.0,
    entropy_max: float = -1,
    entropy_delta: float = 0.1, 
    zero_entropy_threshold: float = 0.2,
    balanced_classes: bool = True,
    seed: int = 42,
) -> None:
    """
    Args:
        precomputed_small_emb_dir: Directory containing embeddings for the small model (generated by precompute_logits.py)
        precomputed_large_emb_dir: Directory containing embeddings for the large model (generated by precompute_logits.py)
        output_dir: Where to save output files
        small_checkpoint_path: The small LM checkpoint path.
        large_checkpoint_path: The large LM checkpoint path.
        entropy_min: The lower bound of the entropy range.
        entropy_max: The upper bound of the entropy range (or -1).
        zero_entropy_threshold: The threshold at which the large model entropy is considered zero.
        balanced_classes: Whether to balance the classes by subsampling the larger class.
    """
    # MUST COME FIRST
    args = locals()

    torch.manual_seed(seed)

    if(entropy_max == -1):
        entropy_max = float("inf")

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
        precomputed_small_emb_dir,
        precomputed_large_emb_dir,
    ]

    logit_loader = PrecomputedShardLoader(shard_dirs)

    filt = {}
    by_label = {}
    small_entropy_dict = {}
    large_entropy_dict = {}
    for i, shard_tups in enumerate(logit_loader):
        if(i % 1000 == 0):
            print(i)
        if (i > 10_000):
            break
            
        small_tup, large_tup = shard_tups

        small_key, small_emb = small_tup
        large_key, large_emb = large_tup

        # Sanity check. The shards should be aligned such that the keys match.
        keys = set([t[0] for t in shard_tups])
        assert(len(keys) == 1)

        small_emb = small_emb.to(device=DEVICE, dtype=DTYPE)
        large_emb = large_emb.to(device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            # Compute logits from the small model embeddings
            small_logits = small_lm_head(small_emb)
            large_logits = large_lm_head(large_emb)

            # Softmax both sets of logits
            small_logits_softmax = torch.nn.functional.softmax(small_logits, dim=-1)
            large_logits_softmax = torch.nn.functional.softmax(large_logits, dim=-1)

            # Compute entropy
            small_logs = torch.nn.functional.log_softmax(small_logits, dim=-1)
            small_entropy = torch.sum(-1 * small_logits_softmax * small_logs, dim=-1)
            large_logs = torch.nn.functional.log_softmax(large_logits, dim=-1)
            large_entropy = torch.sum(-1 * large_logits_softmax * large_logs, dim=-1)

            small_entropy_in_range = torch.logical_and(
                small_entropy >= entropy_min, 
                small_entropy < entropy_max,
            )

            large_entropy_in_range = torch.logical_and(
                large_entropy >= small_entropy - entropy_delta, 
                large_entropy <= small_entropy + entropy_delta,
            )

            large_entropy_zero = large_entropy < zero_entropy_threshold

            # e = epistemic, a = aleatoric
            high_e_low_a = torch.logical_and(
                small_entropy_in_range,
                large_entropy_zero,
            )

            low_e_high_a = torch.logical_and(
                small_entropy_in_range,
                large_entropy_in_range,
            )

            zero_dict = by_label.setdefault("0", {})
            ones_dict = by_label.setdefault("1", {})

            zero_dict[small_key] = high_e_low_a
            ones_dict[small_key] = low_e_high_a

            small_entropy_dict[small_key] = small_entropy
            large_entropy_dict[small_key] = large_entropy
            
    # Balance the classes
    if(balanced_classes):
        sizes = {
            "0": sum([torch.sum(v) for v in by_label["0"].values()]),
            "1": sum([torch.sum(v) for v in by_label["1"].values()]),
        }

        # The class with the most examples
        max_class = max(sizes, key=sizes.get)
        min_class = min(sizes, key=sizes.get)

        fraction = sizes[min_class] / sizes[max_class]
        
        for key, f in by_label[max_class].items():
            subsample_mask = torch.rand(f.shape, device=f.device) <= fraction
            filtered = torch.logical_and(f, subsample_mask)
            filt[key] = torch.logical_or(
                by_label[min_class][key],
                filtered,
            )

            by_label[max_class][key] = filtered

        new_sizes = {
            "0": sum([torch.sum(v) for v in by_label["0"].values()]),
            "1": sum([torch.sum(v) for v in by_label["1"].values()]),
        }

        print(new_sizes)

    filt = {
        k: v.to(device="cpu") for k,v in filt.items()
    }

    output_path = os.path.join(output_dir, "filter.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(filt, fp, protocol=pickle.HIGHEST_PROTOCOL)

    output_path = os.path.join(output_dir, "small_entropy.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(small_entropy_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    output_path = os.path.join(output_dir, "large_entropy.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(large_entropy_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
