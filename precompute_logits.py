import copy
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Optional
import warnings

import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import pipeLLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, jsd


DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')


def main(
    *,
    prompts_json_path: str,
    output_dir: str,
    model_size: str = "30B",
    checkpoint_path: str = None,
    tokenizer_path: Optional[Path] = None,
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
        model_size: The size of the LLAMA model to use. E.g. "7B" or "30B"
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        output_shard_size: Number of outputs per output shard
        return_embeddings: Whether to skip the logit head and return raw embeddings
        return_initial_embeddings: Whether to immediately return the sequence embedding
        resume: Quick and dirty resume functionality. DON'T CHANGE HYPERPARAMS.
    """
    if not checkpoint_path:
        checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth')
    else:
        checkpoint_path = Path(checkpoint_path)
    if not tokenizer_path:
        tokenizer_path = Path('/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model')
    else:
        tokenizer_path = Path(tokenizer_path)
   
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    assert not (return_embeddings and return_initial_embeddings), \
            "Only one of return_embeddings and return_initial_embeddings may be enabled"

    # Create the output dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    # Initialize the model
    print("Loading model... ", file=sys.stderr, end='')
    with EmptyInitOnDevice(
        device=DEVICE, dtype=DTYPE, quantization_mode=quantize
    ):
        t0 = time.time()
#        model = LLaMA.from_name(model_size)
#        checkpoint = torch.load(checkpoint_path)
#        model.load_state_dict(checkpoint, strict=True)
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

    model.eval()

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)

    prompts = list(sorted(prompts.items(), key=lambda t: t[0]))

    def get_shard_path(shard_count):
        output_basename = os.path.split(prompts_json_path)[-1].split('.')[0]
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
        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE)
        len_prompt = len(encoded_prompt)
        encoded_prompt = encoded_prompt.unsqueeze(0)  # add batch dimension

        if(len_prompt > model.config.block_size):
            print(f'Skipping "{key}" (too long)...')
            continue

        # Run the model
        with torch.no_grad():
            fn = model.forward
            if(return_embeddings):
                fn = model._forward
            elif(return_initial_embeddings):
                fn = model.embed_sequence

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
