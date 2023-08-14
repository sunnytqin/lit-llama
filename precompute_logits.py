import copy
import json
import os
import pickle
import sys
import time
from typing import Optional
import warnings

import lightning as L
import torch
from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
)

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import pipeLLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, jsd


DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')
SUPPORTED_MODEL_TYPES = set([
    "llama",
    "pythia",
])


def load_llama(model_size, checkpoint_path, tokenizer_path, quantize):
    assert(os.path.isfile(checkpoint_path))
    assert(os.path.isfile(tokenizer_path))

    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()
    with EmptyInitOnDevice(
        device=DEVICE, dtype=DTYPE, quantization_mode=quantize
    ):
        model = pipeLLaMA.from_name(model_size)
        partition_schedule = model.partition_schedule
        checkpoint = torch.load(checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        model.load_state_dict(checkpoint, strict=True)
    
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = Tokenizer(tokenizer_path)
    tokenizer_fn = lambda p: tokenizer.encode(p, bos=True, eos=False, device=DEVICE)
    
    return model, tokenizer_fn


def load_pythia(model_size, checkpoint_path):
    assert(os.path.isdir(checkpoint_path))

    print("Loading model... ", file=sys.stderr, end='')
    t0 = time.time()

    revisions = os.listdir(checkpoint_path)
    # Revisions are of the format step{number}
    revision = list(sorted(revisions, key=lambda r: int(r.split('step')[-1])))[-1]
    cache_dir = os.path.join(checkpoint_path, revision)

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=DTYPE,
        local_files_only=True,
    )
    model = model.to(device=DEVICE)
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer= AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
    )

    tokenizer_fn = lambda p: (
        tokenizer(p, return_tensors="pt")["input_ids"]
        .squeeze(0)
        .to(device=DEVICE)
    )

    return model, tokenizer_fn


def pythia_forward(model, embeddings=False):
    def fw(input_ids):
        return_dict = model.config.use_return_dict
        outputs = model.gpt_neox(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if(embeddings):
            return hidden_states

        logits = model.embed_out(hidden_states)
        return logits

    return fw


def main(
    *,
    prompts_json_path: str,
    output_dir: str,
    model_type: str = "llama",
    model_size: str = "30B",
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    quantize: Optional[str] = None,
    output_shard_size: int = 2500,
    return_embeddings: bool = False,
    return_initial_embeddings: bool = False,
    resume: bool = False,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompts_json_path: A JSON file containing a dictionary of prompts keyed by prompt IDs
        output_dir: Where to save output pickle files
        model_type: Model class. e.g. "llama" or "pythia"
        model_size: The size of the model to use. E.g. "7B" or "30B"
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit model.
        output_shard_size: Number of outputs per output shard
        return_embeddings: Whether to skip the logit head and return raw embeddings
        return_initial_embeddings: Whether to immediately return the sequence embedding
        resume: Quick and dirty resume functionality. DON'T CHANGE HYPERPARAMS.
    """
    assert(model_type in SUPPORTED_MODEL_TYPES)

    if not checkpoint_path:
        if(model_type == "llama"):
            checkpoint_path = f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth'
        elif(model_type == "pythia"):
            checkpoint_path = f'/n/holystore01/LABS/barak_lab/Everyone/models/pythia/pythia-{model_size}/'
        else:
            raise ValueError
    else:
        checkpoint_path = checkpoint_path
    
    if not tokenizer_path:
        if(model_type == "llama"):
            tokenizer_path = '/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model'
    else:
        tokenizer_path = tokenizer_path

    assert not (return_embeddings and return_initial_embeddings), \
            "Only one of return_embeddings and return_initial_embeddings may be enabled"

    # Create the output dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model and tokenizer
    if(model_type == "llama"):
        model, tokenizer = load_llama(
            model_size, checkpoint_path, tokenizer_path, quantize
        )
    elif(model_type == "pythia"):
        model, tokenizer = load_pythia(
            model_size, checkpoint_path
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.eval()

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)

    prompts = list(sorted(prompts.items(), key=lambda t: t[0]))

    def get_shard_path(shard_count):
        output_basename = os.path.split(prompts_json_path)[-1].split('.')[0]
        output_basename += f"_{model_type}"
        output_basename += f"_{model_size}"
        if(return_embeddings):
            output_basename += "_emb"
        shard_name = f"{output_basename}_{shard_count}.pickle"
        return os.path.join(output_dir, shard_name)

    # Generate logits
    if(resume):
        print(f"Resuming computation...")

    shard_count = 0
    shard_path = get_shard_path(shard_count)
    skip = False
    outputs = {}
    for i, (key, prompt) in enumerate(prompts):
        # Write shard
        if(i != 0 and i % output_shard_size == 0):
            if(len(outputs)):
                with open(shard_path, "wb") as fp:
                    pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"Wrote shard {shard_path}...")

            shard_count += 1
            shard_path = get_shard_path(shard_count)
            outputs = {}

        # Skip precomputed shard entries
        if(resume and os.path.isfile(shard_path)):
            continue

        # Tokenize the prompt
        encoded_prompt = tokenizer(prompt)
        len_prompt = len(encoded_prompt)
        encoded_prompt = encoded_prompt.unsqueeze(0)  # add batch dimension

        if(len_prompt == 0):
            print(f'Skipping "{key}" (too short)...')
            continue

        if((model_type == "llama" and len_prompt > model.config.block_size)):
            print(f'Skipping "{key}" (too long)...')
            continue

        # Run the model
        with torch.no_grad():
            if(model_type == "llama"):
                fn = model.forward
                if(return_embeddings):
                    fn = model._forward
                elif(return_initial_embeddings):
                    fn = model.embed_sequence
            elif(model_type == "pythia"):
                fn = pythia_forward(model)
                if(return_embeddings):
                    fn = pythia_forward(model, embeddings=True)
                elif(return_initial_embeddings):
                    fn = model.gpt_neox.embed_in

            logits = fn(encoded_prompt)

        logits = logits.squeeze(0)
        logits = logits.cpu()
        outputs[key] = logits

    if(len(outputs)):
        with open(shard_path, "wb") as fp:
            pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)


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
