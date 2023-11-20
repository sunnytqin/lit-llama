import argparse
import os
from pathlib import Path

import torch

from lit_llama.utils import jsd
from train_head_utils import PrecomputedShardLoader, load_lm_head


DTYPE = torch.float32
DEVICE = torch.device("cuda:0")

VOCAB_SIZES = {
    "llama": None,
    "pythia": 50254,
}


def compute_entropy(logits):
    logs = torch.log(logits)
    entropy = torch.sum(-1 * logits * logs, dim=-1)
    return entropy


def main(args):
    DATA = []

    model_type = "pythia"
    small_model_size = "1.4b"
    large_model_size = "6.9b"
    small_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/models/pythia/pythia-{small_model_size}"
    large_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/models/pythia/pythia-{large_model_size}"

    #assert("7B" in args.precomputed_small_emb_dir)
    #assert("30B" in args.precomputed_large_emb_dir)
    #model_type = "llama"
    #small_model_size = "7B"
    #large_model_size = "30B"
    #small_checkpoint_path = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/state_dict.pth"
    #large_checkpoint_path = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/state_dict.pth"

    # Load the (small) LM heads of both the small and the large model.
    # We"ve only cached the embeddings and not the (much larger) logits.
    small_lm_head = load_lm_head(small_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=small_model_size)
    large_lm_head = load_lm_head(large_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=large_model_size)

    # Load the precomputed logits
    shard_dirs = [
        args.precomputed_small_emb_dir,
        args.precomputed_large_emb_dir,
    ]

    logit_loader = PrecomputedShardLoader(shard_dirs)

    for i, shard_tups in enumerate(logit_loader):
        small_tup, large_tup = shard_tups

        small_key, small_emb = small_tup
        large_key, large_emb = large_tup

        # Sanity check. The shards should be aligned such that the keys match.
        keys = set([t[0] for t in shard_tups])
        assert(len(keys) == 1)

        # Some empty articles slipped through my filter. Sad!
        if(small_emb.shape[0] == 1):
            continue

        small_emb = small_emb.to(device=DEVICE, dtype=DTYPE)
        large_emb = large_emb.to(device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            # Compute logits from the small model embeddings
            small_logits = small_lm_head(small_emb)
            large_logits = large_lm_head(large_emb)

            vocab_size = VOCAB_SIZES[model_type]
            if(vocab_size):
                small_logits = small_logits[..., :vocab_size]
                large_logits = large_logits[..., :vocab_size]

            small_logits = small_logits.double()
            large_logits = large_logits.double()

            # Softmax both sets of logits
            small_logits_softmax = torch.nn.functional.softmax(small_logits, dim=-1)
            large_logits_softmax = torch.nn.functional.softmax(large_logits, dim=-1)

            small_entropy = compute_entropy(small_logits_softmax)
            large_entropy = compute_entropy(large_logits_softmax)
            jsd_vals = jsd(small_logits, large_logits)

            for s, l, j in zip(torch.unbind(small_entropy), torch.unbind(large_entropy), torch.unbind(jsd_vals)):
                DATA.append((s.item(), l.item(), j.item()))

    print(len(DATA))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "embedding_comparison.txt"), "w") as f:
        for t in DATA:
            joint = "\t".join([str(x) for x in t])
            f.write(f"{joint}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect embedding data.")
    parser.add_argument("precomputed_small_emb_dir", type=str, help="Path to the directory containing the precomputed embeddings for the small model.")
    parser.add_argument("precomputed_large_emb_dir", type=str, help="Path to the directory containing the precomputed embeddings for the large model.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the output will be saved.")
    args = parser.parse_args()

    main(args)
