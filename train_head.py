import copy
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Optional, Iterator, Tuple
import warnings

import lightning as L
import torch
import torch.nn as nn
import wandb

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import pipeLLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, jsd


DTYPE = torch.float32
DEVICE = torch.device("cuda:0")
# WandB x axis
WANDB_STEP_METRICS = set(["step", "epoch"])
# WandB y, x pairs
wandb_metrics = set([
    ("train_loss", "step"),
    ("train_accuracy", "step"),
    ("val_loss", "step"),
    ("val_accuracy", "step"),
])


class DistancePredictionHead(nn.Module):
    def __init__(self,
        input_dim: int,
        no_bins: int,
        hidden_dim: int,
        no_hidden_layers: int,
        dropout: float,
        log_scale: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.no_bins = no_bins
        self.hidden_dim = hidden_dim
        self.no_hidden_layers = no_hidden_layers
        self.dropout = dropout
        self.log_scale = log_scale

        if activation == "relu":
            activation_class = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()

        if(no_hidden_layers == 0):
            self.layers.append(nn.Linear(input_dim, no_bins))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation_class())
            for _ in range(no_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(activation_class())

            self.layers.append(nn.Linear(hidden_dim, no_bins))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class PrecomputedShardLoader:
    def __init__(self, 
        shard_dirs: list[str],
    ):
        """
            Loads shards generated by precompute_logits.py and yields examples one by one.

            Args:
                shard_dirs: 
                    List of directories containing (only) sets of 
                    corresponding shards computed by precompute_logits.py
                shuffle_shards: 
                    Whether to shuffle the shards
                shuffle_seed: 
                    Seed for shuffling the shards
        """
        self.shard_dirs = shard_dirs

        shard_name_lists = []
        for shard_dir in self.shard_dirs:
            shards = os.listdir(shard_dir)

            # Shard names are assumed to be in the format "name_number.pickle"
            shards = list(sorted(shards, key=lambda x: int(x.split('_')[-1].strip('.pickle'))))

            shard_name_lists.append(shards)

        l = len(shard_name_lists[0])
        assert(all([len(shard_name_list) == l for shard_name_list in shard_name_lists]))

        shards = list(zip(*shard_name_lists))
        self.shards = shards

    def load_shards(self, shard_id: int):
        shards = []
        for shard_dir, shard_name in zip(self.shard_dirs, self.shards[shard_id]):
            shard_path = os.path.join(shard_dir, shard_name)
            with open(shard_path, "rb") as fp:
                shard = pickle.load(fp)

            shards.append(shard)

        return shards

    def shuffle_shards(self, seed: int):
        random.Random(seed).shuffle(self.shards)

    def __iter__(self):
        return self._gen()

    def _gen(self):
        """
            Returns a generator that yields tuples of examples one by one.
        """
        cur_shard_id = 0
        while cur_shard_id < len(self.shards):
            t = time.time()
            # Load corresponding shards
            t = time.time()
            logging.info(f"Loading shards...")
            loaded_shards = self.load_shards(cur_shard_id)
            logging.info(f"Shards loaded ({time.time() - t:.02f} seconds)...")

            # All shards in the tuple should be of the same length
            shard_len = len(loaded_shards[0])
            assert(all([len(shard) == shard_len for shard in loaded_shards]))

            # Sort examples within each shard by key
            sort_shard = lambda l: list(sorted(l.items(), key=lambda t: t[0]))
            for i in range(len(loaded_shards)):
                loaded_shards[i] = sort_shard(loaded_shards[i])
            
            yield from zip(*loaded_shards)

            cur_shard_id += 1

            del loaded_shards


def load_lm_head(checkpoint_path: str):
    # Load the small model's LM head
    logging.info(f"Loading model at {checkpoint_path}... ")
    t = time.time()
    checkpoint = torch.load(checkpoint_path)
    assert(len([k for k in checkpoint.keys() if "lm_head" in k]) == 1)
    lm_head_weights = checkpoint["lm_head.weight"]
    vocab_size, emb_dim = lm_head_weights.shape
    lm_head = nn.Linear(
        emb_dim, vocab_size, bias=False
    )
    with torch.no_grad():
        lm_head.weight.data = lm_head_weights.to(DTYPE)
        lm_head.eval()
        lm_head = lm_head.to(DEVICE)

    logging.info(f"Time: {time.time() - t:.02f} seconds.")

    del checkpoint

    return lm_head


def _discretize(
    values: torch.Tensor, 
    no_bins: int, 
    mi: float, 
    ma: float
):
    """
        Discretizes the target into `no_bins` bins.
    """
    assert(mi < ma)
    assert(no_bins > 0)

    # Clamp the values to the range [mi, ma]
    values = torch.clamp(values, min=mi, max=ma)

    boundaries = torch.linspace(
        mi, ma, no_bins + 1, device=values.device
    )
    boundaries[..., -1] = float('inf')

    # Make shapes compatible
    boundaries = boundaries.view(*([1]*len(values.shape)), -1)
    values = values.unsqueeze(-1)

    lt = boundaries[..., :-1] <= values
    gt = boundaries[..., 1:] > values
    bin_id = torch.logical_and(lt, gt).to(torch.int64).argmax(dim=-1)
    
    return bin_id


def _preprocessor(
    shard_loader: PrecomputedShardLoader,
    small_lm_head: nn.Linear,
    large_lm_head: nn.Linear,
    no_bins: int,
    min_bin: float,
    max_bin: float,
    device: torch.device,
):
    for i, (small_tup, large_tup) in enumerate(shard_loader):
        small_key, small_emb = small_tup
        large_key, large_emb = large_tup

        # Sanity check. The shards should be aligned such that the keys match.
        assert(small_key == large_key)

        # Some empty articles slipped through my filter. Sad!
        if(small_emb.shape[0] == 1):
            continue

        small_emb = small_emb.to(device=device, dtype=DTYPE)
        large_emb = large_emb.to(device=device, dtype=DTYPE)
        
        with torch.no_grad():
            # Compute logits from the small model embeddings
            small_logits = small_lm_head(small_emb)
            large_logits = large_lm_head(large_emb)

            # Softmax both sets of logits
            small_logits = torch.nn.functional.softmax(small_logits, dim=-1)
            large_logits = torch.nn.functional.softmax(large_logits, dim=-1)

            # Compute the JS divergence between the two distributions
            divergence = jsd(small_logits, large_logits)
            
            # We will predict the log of the divergence
            target = torch.log(divergence)

            # Discretize the target
            target = _discretize(
                target,
                no_bins, 
                mi=min_bin, 
                ma=max_bin,
            ).squeeze(0)

        yield (small_emb, target)


def batch_loader(
    data_gen: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    skip_frac: float,
):
    batch = []
    for i, (small_emb, log_js_div) in enumerate(data_gen):
        # [N, emb_dim]
        assert(len(small_emb.shape) == 2)
        inputs = torch.unbind(small_emb, dim=-2)

        # [N]
        assert(len(log_js_div.shape) == 1)
        targets = torch.unbind(log_js_div, dim=-1)

        assert(len(inputs) == len(targets))

        for inp, target in zip(inputs, targets):
            # We don't want too many consecutive tokens from the same prompt,
            # so we skip a large percentage of them.
            if(random.random() < skip_frac):
                continue

            batch.append((inp, target))

            if(len(batch) == batch_size):
                inputs = torch.stack([t[0] for t in batch])
                targets = torch.stack([t[1] for t in batch])

                assert(inputs.device == targets.device)

                yield inputs, targets

                batch = []

    # Toss the last batch if it's too small
    pass

def _wandb_setup(args):
    wandb_args = {
        "project": "wandb_project",
        "entity": "wandb_entity",
        "name": "wandb_run_name",
        "dir": "output_dir",
    }
    for arg in wandb_args.values():
        if(args[arg] is None):
            raise ValueError(f"Must provide {arg} if use_wandb is True")

    wandb.login()

    wandb_run = wandb.init(
        config=args,
        **{k:args[v] for k, v in wandb_args.items()},
    )

    for step_metric in WANDB_STEP_METRICS:
        wandb.define_metric(step_metric)

    for metric, step_metric in wandb_metrics:
        assert(step_metric in WANDB_STEP_METRICS)
        wandb.define_metric(metric, step_metric=step_metric)

    # Save the git diff for reproducibility
    git_diff_path = os.path.join(args[wandb_args["dir"]], "git_diff.txt")
    os.system(f"git diff > {git_diff_path}")
    wandb.save(git_diff_path, base_path=f"./{args[wandb_args['dir']]}")

    return wandb_run


def _wandb_log(metrics, step_metric, step):
    assert step_metric in WANDB_STEP_METRICS
    for metric in metrics:
        assert (metric, step_metric) in wandb_metrics, \
            f"Metric {metric} not defined in the `metrics' dict"

    metrics = {
        **metrics,
        step_metric: step,
    }

    wandb.log(metrics)


def main(
    *,
    precomputed_small_emb_dir: str,
    precomputed_large_emb_dir: str,
    output_dir: str,
    model_size: str = "7B",
    small_checkpoint_path: str = None,
    large_checkpoint_path: str = None,
    hidden_dim: int = 2048,
    no_hidden_layers: int = 5,
    dropout: float = 0.1,
    activation: str = "relu",
    lr: float = 1e-6,
    batch_size: int = 64,
    no_epochs: int = 5,
    skip_frac: float = 0.95,
    no_bins: int = 2,
    min_bin: float = -3.,
    max_bin: float = 0.,
    seed: int = 42,
    precomputed_small_emb_dir_val: str = None,
    precomputed_large_emb_dir_val: str = None,
    eval_every_n_batches: int = 1000,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_name: str = None,
) -> None:
    """
    Args:
        precomputed_small_emb_dir: Directory containing embeddings for the small model (generated by precompute_logits.py)
        precomputed_large_emb_dir: Directory containing embeddings for the large model (generated by precompute_logits.py)
        output_dir: Where to save output files
        model_size: The size of the SMALL model. E.g. "7B" or "30B"
        checkpoint_path: The small LM checkpoint path.
        hidden_dim: Hidden dimension of the distance prediction head.
        no_hidden_layers: Number of hidden layers in the distance prediction head.
        dropout: Dropout probability in the distance prediction head.
        activation: Activation function in the distance prediction head.
        lr: Learning rate.
        batch_size: Batch size.
        no_epochs: Number of epochs.
        skip_frac: Probability of skipping any given token.
        no_bins: Number of bins to discretize the target into.
        min_bin: Minimum value of the discretized target.
        max_bin: Maximum value of the discretized target.
        seed: Random seed.
        precomputed_small_emb_dir_val: Directory containing validation prompt embeddings for the small model (generated by precompute_logits.py)
        precomputed_large_emb_dir_val: Directory containing validation prompt embeddings for the large model (generated by precompute_logits.py)
        eval_every_n_batches: How often validation is performed (in batches)
        use_wandb: Whether to upload logs to Weights and Biases
        wandb_project: Weights and Biases project name. Mandatory with use_wandb.
        wandb_entity: Weights and Biases entity name. Mandatory with use_wandb.
        wandb_run_name: Weights and Biases run name. Mandatory with use_wandb.
    """
    # MUST COME FIRST
    args = locals()

    torch.manual_seed(seed)
    random.seed(seed)

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)

    if(use_wandb):
        # Add some metrics that depend on arguments
        wandb_metrics.add(
            (f"val_confusion_matrix_{no_bins}", "step")
        )
        for i in range(no_bins):
            for j in range(no_bins):
                wandb_metrics.add(
                    (f"val_confusion_matrix_{no_bins}_{i}_{j}", "step")
                )

        # Init WandB, register metrics, etc.
        _wandb_setup(args)

    if not small_checkpoint_path:
        small_checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/state_dict.pth')
    if not large_checkpoint_path:
        large_checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/state_dict.pth')

    assert small_checkpoint_path.is_file()
    assert large_checkpoint_path.is_file()

    # Load the (small) LM heads of both the small and the large model.
    # We've only cached the embeddings and not the (much larger) logits.
    small_lm_head = load_lm_head(small_checkpoint_path)
    large_lm_head = load_lm_head(large_checkpoint_path)

    # Load the precomputed logits
    logit_loader = PrecomputedShardLoader(
        [
            precomputed_small_emb_dir,
            precomputed_large_emb_dir
        ],
    )

    val = precomputed_small_emb_dir_val is not None
    if(val and precomputed_large_emb_dir_val is None):
        raise ValueError("Must provide both small and large validation directories")

    if(val):
        logging.info("Validation enabled...")
    else:
        logging.warning("Validation disabled...")

    val_logit_loader = None
    if(val):
        val_logit_loader = PrecomputedShardLoader(
            [
                precomputed_small_emb_dir_val,
                precomputed_large_emb_dir_val,
            ],
        )

    # Initialize the model
    distance_prediction_head = DistancePredictionHead(
        input_dim=small_lm_head.weight.shape[1],
        no_bins=no_bins,
        hidden_dim=hidden_dim,
        no_hidden_layers=no_hidden_layers,
        dropout=dropout,
        activation=activation,
    )
    distance_prediction_head.to(DEVICE)
    param_count = sum(
        p.numel() for p in distance_prediction_head.parameters() if p.requires_grad
    )
    logging.info(f"Loaded prediction head ({param_count} parameters)...")

    # Umm uhh
    distance_prediction_head = torch.compile(distance_prediction_head)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        distance_prediction_head.parameters(), 
        lr=lr
    )

    # Select the loss function
    loss_fn = torch.nn.functional.cross_entropy

    # Standard training loop
    cum_step = 0
    for epoch in range(no_epochs):
        if(use_wandb):
            wandb.log({"epoch": epoch})

        shuffle_seed = random.randint(0, 2**32 - 1)
        logit_loader.shuffle_shards(shuffle_seed)

        data_gen = _preprocessor(
            shard_loader=logit_loader,
            small_lm_head=small_lm_head,
            large_lm_head=large_lm_head,
            no_bins=no_bins,
            min_bin=min_bin,
            max_bin=max_bin,
            device=DEVICE,
        )

        bl = batch_loader(
            data_gen=data_gen,
            batch_size=batch_size,
            skip_frac=skip_frac,
        )

        for i, (inputs, targets) in enumerate(bl):
            # Dry run w/ grad enabled for the torch compiler (idk why this is necessary)
            if(i == 0 and epoch == 0):
                distance_prediction_head(inputs)

            # Periodically run the validation loop
            if(val and i % eval_every_n_batches == 0):
                val_data_gen = _preprocessor(
                    shard_loader=val_logit_loader,
                    small_lm_head=small_lm_head,
                    large_lm_head=large_lm_head,
                    no_bins=no_bins,
                    min_bin=min_bin,
                    max_bin=max_bin,
                    device=DEVICE,
                )

                val_bl = batch_loader(
                    data_gen=val_data_gen,
                    batch_size=batch_size,
                    skip_frac=0.,
                )

                with torch.no_grad():
                    val_loss_sum = 0
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
                        val_loss = loss_fn(val_outputs, val_targets)
                        val_loss_sum += torch.sum(val_loss).item()

                        val_preds = torch.argmax(val_outputs, dim=-1)

                        val_acc = val_preds == val_targets
                        val_acc_sum += torch.mean(val_acc.float()).item()

                        val_batch_count += 1

                        all_val_preds.extend(val_preds.cpu().tolist())
                        all_val_gt.extend(val_targets.cpu().tolist())

                    confusion_matrix = torch.zeros(no_bins, no_bins)
                    for gt, pred in zip(all_val_gt, all_val_preds):
                        confusion_matrix[gt, pred] += 1

                    confusion_matrix = confusion_matrix / (confusion_matrix.sum() + 1e-6)

                    val_metrics = {
                        "val_loss": val_loss_sum / val_batch_count,
                        "val_accuracy": val_acc_sum / val_batch_count,
                        f"val_confusion_matrix_{no_bins}": confusion_matrix,
                    }

                    for k,v in val_metrics.items():
                        print(f"Validation metric {k}: {v}")

                    if(use_wandb):
                        # Make a nice WandB confusion matrix
                        val_wandb_metrics = val_metrics

                        val_wandb_metrics[f"val_confusion_matrix_{no_bins}"] = wandb.plot.confusion_matrix(
                            y_true=all_val_gt,
                            preds=all_val_preds,
                            class_names=[str(label) for label in range(no_bins)],
                        )

                        # Annoyingly, WandB doesn't support plotting confusion matrices over time
                        # We need to add those values separately
                        for row in range(no_bins):
                            for col in range(no_bins):
                                val_wandb_metrics[f"val_confusion_matrix_{no_bins}_{row}_{col}"] = (
                                    confusion_matrix[row,col]
                                )

                        _wandb_log(metrics=val_wandb_metrics, step_metric="step", step=cum_step)

            inputs = inputs.to(device=DEVICE, dtype=DTYPE)
            targets = targets.to(device=DEVICE, dtype=torch.int64)

            optimizer.zero_grad()
            outputs = distance_prediction_head(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            accuracy = torch.sum(
                torch.argmax(outputs, dim=-1) == targets
            ) / targets.numel()

            metrics = {
                "train_loss": loss.item(),
                "train_accuracy": accuracy.item(),
            }

            if(use_wandb):
                _wandb_log(metrics=metrics, step_metric="step", step=cum_step)

            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i}, loss: {loss.item():.02f}", file=sys.stderr)
                print(f"Epoch {epoch}, batch {i}, accuracy: {accuracy.item():.02f}", file=sys.stderr)

            cum_step += 1

    # Save the model
    model_path = os.path.join(output_dir, "state_dict.pth")
    torch.save(distance_prediction_head.state_dict(), model_path)


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
