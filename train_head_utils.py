import logging
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Optional, Iterator, Tuple, Sequence

import lightning
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    GPTNeoXForCausalLM,
)

import lit_gpt
from lit_llama import LLaMA, Tokenizer, pipeLLaMA
from lit_llama.utils import EmptyInitOnDevice, jsd


DTYPE = torch.float32
DEVICE = torch.device("cuda:0")

VOCAB_SIZES = {
    "llama": None,
    "llama_2": None,
    "pythia": 50254,
}

MAX_LEN = 2048


class PrecomputedShardLoader:
    def __init__(self, 
        shard_dirs: Sequence[str],
        dataset_filter_path: Optional[str] = None,
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

        self.filter = None
        if(dataset_filter_path):
            with open(dataset_filter_path, "rb") as fp:
                self.filter = pickle.load(fp)

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
            t = time.time()
            print(f"Loading shard {shard_path}...")
            with open(shard_path, "rb") as fp:
                shard = pickle.load(fp)
            print(f"Loading done ({time.time() - t}s)...")
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

            if(self.filter):
                # Filter out examples that don't pass the filter
                for i in range(len(loaded_shards)):
                    shard = loaded_shards[i]
                    y = 0
                    for j in range(len(shard)):
                        k, v = shard[j]
                        
                        # Skip empty articles
                        if(len(v.shape) == 1):
                            continue

                        v = v[:MAX_LEN]
                        shard[j] = (k, v[self.filter[k].bool()])

            yield from zip(*loaded_shards)

            cur_shard_id += 1

            del loaded_shards


def load_llama_tokenizer(tokenizer_path, device, return_tokenizer_as_fn=True):
    tokenizer = Tokenizer(tokenizer_path)
    if(return_tokenizer_as_fn):
        def tokenizer_fn(p):
            return tokenizer.encode(p, bos=True, eos=False, device=device)
        return tokenizer_fn
    return tokenizer


def load_llama(model_size, checkpoint_path, tokenizer_path, dtype, quantize, return_tokenizer_as_fn=True):
    assert(os.path.isfile(checkpoint_path))
    assert(os.path.isfile(tokenizer_path))

    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()
    with EmptyInitOnDevice(
        device=DEVICE, dtype=dtype, quantization_mode=quantize
    ):
        model = pipeLLaMA.from_name(model_size)
        # model = LLaMA.from_name(model_size)
        partition_schedule = model.partition_schedule
        checkpoint = torch.load(checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=True)
    
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = load_llama_tokenizer(
        tokenizer_path, DEVICE, return_tokenizer_as_fn=return_tokenizer_as_fn
    )
    
    return model, tokenizer


def load_llama_2_tokenizer(tokenizer_path, device, return_tokenizer_as_fn=True):
    if(type(tokenizer_path) == str):
        tokenizer_path = pathlib.Path(tokenizer_path) 
    
    tokenizer = lit_gpt.Tokenizer(tokenizer_path)
    if(return_tokenizer_as_fn):
        return lambda p: tokenizer.encode(p, device=device)

    return tokenizer


def load_llama_2(model_size, checkpoint_dir, dtype, return_tokenizer_as_fn=True, return_fabric=False):
    assert(os.path.isdir(checkpoint_dir))

    checkpoint_dir = pathlib.Path(checkpoint_dir)

    if(dtype == torch.float):
        precision = "32"
    elif(dtype == torch.bfloat16):
        precision = "bf16-true"
    else:
        raise ValueError

    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()

    strategy = lightning.fabric.strategies.FSDPStrategy(
        auto_wrap_policy={lit_gpt.model.Block}, cpu_offload=False
    )
    fabric = lightning.Fabric(
        devices=torch.cuda.device_count(), 
        precision=precision, 
        strategy=strategy,
    )
    fabric.launch()
   
    lit_gpt.utils.check_valid_checkpoint_dir(checkpoint_dir)
    config = lit_gpt.Config.from_json(
        checkpoint_dir / "lit_config.json"
    )

    with fabric.init_module(empty_init=True), lit_gpt.utils.quantization(None):
        model = lit_gpt.GPT(config)
   
    model.eval()
    model = fabric.setup_module(model)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    lit_gpt.utils.load_checkpoint(fabric, model, checkpoint_path, strict=True)

    print("Model loaded...", file=sys.stderr)

    tokenizer = load_llama_2_tokenizer(
        checkpoint_dir,
        device=fabric.device,
        return_tokenizer_as_fn=return_tokenizer_as_fn,
    )

    if(return_fabric):
        return model, tokenizer, fabric

    return model, tokenizer


def load_pythia_model(checkpoint_path: str, model_size: str, dtype: torch.dtype, revision: int = -1):
    # Revisions are of the format step{number}
    if(revision == -1):
        revisions = os.listdir(checkpoint_path)
        revision = list(sorted(revisions, key=lambda r: int(r.split('step')[-1])))[-1]
    else:
        revision = f"step{revision}"

    cache_dir = os.path.join(checkpoint_path, revision)
    print("Pythia revision:", revision, "cache_dir:", cache_dir)

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        local_files_only=False,
    )

    return model


def load_pythia_tokenizer(model_size, device):
    tokenizer= AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
    )

    tokenizer_fn = lambda p: (
        tokenizer(p, return_tensors="pt", add_special_tokens=True)["input_ids"]
        .squeeze(0)
        .to(device=DEVICE)
    )

    return tokenizer_fn


def load_pythia(model_size, checkpoint_path, dtype, revision=-1):
    assert(os.path.isdir(checkpoint_path))

    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()

    model = load_pythia_model(
        checkpoint_path, 
        model_size, 
        dtype, 
        revision=revision
    )
    model = model.to(device=DEVICE)
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = load_pythia_tokenizer(model_size, DEVICE)

    return model, tokenizer

def load_icml():
    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()
    model = LlamaForCausalLM.from_pretrained("swj0419/7b_iclm").to(DEVICE)
    model = model.to(device=DEVICE)
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    tokenizer_path = Path('/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model')
    tokenizer = load_llama_tokenizer(
        tokenizer_path, DEVICE, return_tokenizer_as_fn=True
    )

    return model, tokenizer


def load_lm_head(
    checkpoint_path: str, 
    dtype: torch.dtype, 
    device: str, 
    model_type: str,
    model_size: str,
    revision: int = -1,
):
    logging.info(f"Loading model at {checkpoint_path}... ")
    t = time.time()
    if(model_type == "llama"): 
        assert(os.path.isfile(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        assert(len([k for k in checkpoint.keys() if "lm_head" in k]) == 1)
        lm_head_weights = checkpoint["lm_head.weight"]
        vocab_size, emb_dim = lm_head_weights.shape
        lm_head = nn.Linear(
            emb_dim, vocab_size, bias=False
        )
        with torch.no_grad():
            lm_head.weight.data = lm_head_weights.to(dtype)
            lm_head.eval()
            lm_head = lm_head.to(device)

        del checkpoint
    elif(model_type == "llama_2"):
        assert(os.path.isdir(checkpoint_path))
        checkpoint = torch.load(f"{checkpoint_path}/lit_model.pth")
        assert(len([k for k in checkpoint.keys() if "lm_head" in k]) == 1)
        lm_head_weights = checkpoint["lm_head.weight"]
        vocab_size, emb_dim = lm_head_weights.shape
        lm_head = nn.Linear(
            emb_dim, vocab_size, bias=False
        )
        with torch.no_grad():
            lm_head.weight.data = lm_head_weights.to(dtype)
            lm_head.eval()
            lm_head = lm_head.to(device)

        del checkpoint
    elif(model_type == "pythia"):
        assert(os.path.isdir(checkpoint_path))
        model = load_pythia_model(checkpoint_path, model_size, dtype, revision=revision)
        lm_head = model.embed_out
        lm_head = lm_head.eval()
        lm_head = lm_head.to(device)

        del model
    elif(model_type == "icml"):
        model, _ = load_icml()
        lm_head = model.lm_head
        lm_head = lm_head.eval()
        lm_head = lm_head.to(device)

        del model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logging.info(f"Time: {time.time() - t:.02f} seconds.")

    return lm_head


def load_embedding_layer(
    checkpoint_path: str, 
    dtype: torch.dtype, 
    device: str, 
    model_type: str,
    model_size: str,
    revision: int = -1,
):
    logging.info(f"Loading model at {checkpoint_path}... ")
    t = time.time()
    if(model_type == "llama"): 
        assert(os.path.isfile(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        embed_layer_weights = checkpoint["transformer.wte.weight"]
        vocab_size, emb_dim = embed_layer_weights.shape
        emb_layer = nn.Embedding(
            vocab_size, emb_dim,
        )
        with torch.no_grad():
            emb_layer.weight.data = embed_layer_weights.to(dtype)
            emb_layer.eval()
            emb_layer = emb_layer.to(device)

        del checkpoint
    elif(model_type == "llama_2"):
        raise NotImplementedError
    elif(model_type == "pythia"):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logging.info(f"Time: {time.time() - t:.02f} seconds.")

    return emb_layer


class DistancePredictionHeadWithLMHead(nn.Module):
    def __init__(self,
        lm_head: nn.Linear,
        no_bins: int,
        hidden_dim: int,
        no_hidden_layers: int,
        dropout: float,
        log_scale: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = lm_head.weight.shape[1]
        self.token_dim = lm_head.weight.shape[0]
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

        has_bias = lm_head.bias is not None
        local_lm_head = nn.Linear(self.input_dim, self.token_dim, bias=has_bias)
        with torch.no_grad():
            local_lm_head.weight.copy_(lm_head.weight)
            if(has_bias):
                local_lm_head.bias.copy_(lm_head.bias)

        self.layers.append(local_lm_head)

        if(no_hidden_layers == 0):
            self.layers.append(nn.Linear(self.token_dim, no_bins))
        else:
            self.layers.append(nn.Linear(self.token_dim, hidden_dim))
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


def discretize(
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
    model_type: str,
    no_bins: int,
    min_bin: float,
    max_bin: float,
    min_entropy: float,
    max_entropy: float,
    provide_entropy_as_input: bool,
    use_logits_as_input: bool,
    softmax_input_logits: bool,
    device: torch.device,
    dtype: torch.dtype,
    target_fn_name="log_jsd",
    bin_target: bool = True,
    append_predicted_token_embedding=False,
    small_embedding_layer=None,
    _stash=None,
):
    _stash_contents = {}
    for i, shard_tups in enumerate(shard_loader):
        small_tup, large_tup = shard_tups[:2]

        small_key, small_emb = small_tup
        large_key, large_emb = large_tup

        if(len(shard_tups) == 3):
            input_key, input_emb = shard_tups[2]
        elif(len(shard_tups) == 2):
            input_key, input_emb = small_tup
        else:
            raise ValueError("Something went wrong...")

        # Sanity check. The shards should be aligned such that the keys match.
        keys = set([t[0] for t in shard_tups])
        assert(len(keys) == 1)

        # Some empty articles slipped through my filter. Sad!
        if(small_emb.shape[0] == 0):
            continue

        small_emb = small_emb.to(device=device, dtype=dtype)
        large_emb = large_emb.to(device=device, dtype=dtype)
        input_emb = input_emb.to(device=device, dtype=dtype)

        with torch.no_grad():
            # Compute logits from the small model embeddings
            small_logits = small_lm_head(small_emb)
            large_logits = large_lm_head(large_emb)

            # Pythia models inexplicably use different amounts of padding at different sizes
            vocab_size = VOCAB_SIZES[model_type]
            if(vocab_size):
                small_logits = small_logits[..., :vocab_size]
                large_logits = large_logits[..., :vocab_size]

            # Softmax both sets of logits
            small_logits_softmax = torch.nn.functional.softmax(small_logits, dim=-1)
            large_logits_softmax = torch.nn.functional.softmax(large_logits, dim=-1)

            if(softmax_input_logits and use_logits_as_input):
                input_emb = small_logits_softmax
            elif(use_logits_as_input):
                input_emb = small_logits

            small_logs = torch.nn.functional.log_softmax(small_logits, dim=-1)
            small_entropy = torch.sum(-1 * small_logits_softmax * small_logs, dim=-1)
            large_logs = torch.nn.functional.log_softmax(large_logits, dim=-1)
            large_entropy = torch.sum(-1 * large_logits_softmax * large_logs, dim=-1)

            if((min_entropy is not None) and (max_entropy is not None)):
                filt = torch.logical_and(
                    small_entropy >= min_entropy,
                    small_entropy < max_entropy,
                )
            elif((min_entropy is None) ^ (max_entropy is None)):
                raise ValueError("Either none or both of min_entropy and max_entropy must be specified")
            else:
                filt = torch.ones_like(small_entropy)

            # FILTER DONE #
            ###############

            def _unpack(tensor):
                return [t.item() for t in tensor.cpu().unbind()]
            
            def apply_filter(tensor, filter_tensor):
                filtered = [t for f, t in zip(filter_tensor, tensor) if f]
                return torch.stack(filtered) if len(filtered) else tensor[:0]

            _stash_contents.setdefault("small_entropy", []).extend(
                _unpack(apply_filter(small_entropy, filt))
            )
            _stash_contents.setdefault("large_entropy", []).extend(
                _unpack(apply_filter(large_entropy, filt))
            )

            # Compute the target
            if(target_fn_name == "log_jsd"):
                divergence = jsd(small_logits, large_logits)

                # Sometimes precision errors cause divergence to be negative
                divergence = torch.clamp(divergence, min=1e-8)

                # We will predict the log of the divergence
                target = torch.log(divergence)

                _stash_contents.setdefault("divergence", []).extend(
                    _unpack(apply_filter(divergence, filt))
                )
            elif(target_fn_name == "jsd"):
                divergence = jsd(small_logits, large_logits)
                target = divergence

                _stash_contents.setdefault("divergence", []).extend(
                    _unpack(apply_filter(divergence, filt))
                )
            elif(target_fn_name == "small_entropy"):
                target = small_entropy
            elif(target_fn_name == "large_entropy"):
                target = large_entropy
            elif(target_fn_name == "log_large_entropy"):
                clamped_large_entropy = torch.clamp(large_entropy, min=5e-2, max=10)
                target = torch.log(clamped_large_entropy)
            elif(target_fn_name == "large_logits"):
                target = large_logits_softmax
                if(vocab_size):
                    target = target[..., :vocab_size]
            else:
                raise ValueError("Invalid target name")

            if(bin_target):
                # Discretize the target
                target = discretize(
                    target,
                    no_bins, 
                    mi=min_bin, 
                    ma=max_bin,
                )

        input_emb = apply_filter(input_emb, filt)
        if(input_emb.shape[0] == 0):
            continue

        target = apply_filter(target, filt)

        if(provide_entropy_as_input):
            entropy_filtered = apply_filter(small_entropy, filt)
            input_emb = torch.cat([input_emb, entropy_filtered.unsqueeze(-1)], dim=-1)

        if(append_predicted_token_embedding):
            small_logits_softmax_filtered = apply_filter(small_logits_softmax, filt)
            small_top_1_token = torch.argmax(small_logits_softmax_filtered, dim=-1)
            small_top_1_token_emb = small_embedding_layer(small_top_1_token).detach() # for some reason .eval() isn't sufficient for these
            input_emb = torch.cat(
                [
                    input_emb,
                    small_top_1_token_emb,
                ],
                dim=-1,
            )

        yield (input_emb, target)

    if(_stash is not None):
        _stash.update(_stash_contents)


def batch_loader(
    data_gen: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    skip_frac: float,
    nonzero_bin_weight: float = 1.,
):
    def _package_batch(batch):
        inputs = torch.stack([t[0] for t in batch])
        targets = torch.stack([t[1] for t in batch])

        assert(inputs.device == targets.device)

        return inputs, targets

    batch = []
    for i, (small_emb, target) in enumerate(data_gen):
        # [N, emb_dim]
        assert(len(small_emb.shape) == 2)
        inputs = torch.unbind(small_emb, dim=-2)

        # [N]
        target_is_categorical = len(target.shape) == 1
        tdim = -1 if target_is_categorical else -2
        targets = torch.unbind(target, dim=tdim)

        assert(len(inputs) == len(targets))

        for inp, target in zip(inputs, targets):
            if(target_is_categorical):
                weighted_skip_frac = skip_frac / (nonzero_bin_weight if target != 0 else 1.)

                # We don't want too many consecutive tokens from the same prompt,
                # so we skip a large percentage of them.
                if(random.random() < weighted_skip_frac):
                    continue
            else:
                assert(skip_frac == 0)

            batch.append((inp, target))

            if(len(batch) == batch_size):
                yield _package_batch(batch)
                batch = []

    # Serve the final batch, even if it's not full
    if(len(batch) > 0):
        yield _package_batch(batch)


def entropy_threshold_acc(small_entropy, ground_truth):
    sorted_pairs = list(sorted(zip(small_entropy, ground_truth), key=lambda x: x[0]))

    ones_count = 0
    ones_so_far = []
    for s, gt in sorted_pairs:
        if(gt == 1):
            ones_count += 1
        
        ones_so_far.append(ones_count)

    total_ones = ones_so_far[-1]
    best_acc = -1
    best_index = -1
    for i in range(len(sorted_pairs)):
        zeros_correct = i - ones_so_far[i] + 1
        ones_correct = total_ones - ones_so_far[i]
        acc = (zeros_correct + ones_correct) / len(sorted_pairs)
        if(acc > best_acc):
            best_acc = acc
            best_index = i

    return best_acc
