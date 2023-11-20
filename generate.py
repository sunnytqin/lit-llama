import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA, LLaMA
from lit_llama.utils import EmptyInitOnDevice
from pathlib import Path

from repetition import log, repetition_experiment

DEVICE= "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# DTYPE = torch.float32



def main(
    model_type: str = "llama", 
    model_size: str = "7B",
    k: int = 10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,
    sample_until_period: bool = True,
    addl_token_limit: int = 100,
):
    """
    Args:
        model_type: "llama", 
        model_size: [7B, 30B],
        k: top k token for repetition,
        tokenizer_path: llama tokenizer,
        checkpoint_path: model checkpoint default if None,
        sample_until_period: whether sample until period,
        addl_token_limit: sample until period hard cutoff,
    """

    if(checkpoint_path is None):
        checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth")

    if(tokenizer_path is None):
        tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if model_size == "7B":
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model ...", end='', file=sys.stderr)
            t0 = time.time()
            model = LLaMA.from_name(model_size)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    else: 
        # Initialize the model
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model... ", end='')
            t0 = time.time()
            model = pipeLLaMA.from_name(model_size)
            partition_schedule = model.partition_schedule
            checkpoint = torch.load(checkpoint_path)
            for key in list(checkpoint.keys()):
                if 'transformer.h' in key:
                    split = key.split('.')
                    split[2] = partition_schedule[int(split[2])]
                    checkpoint[".".join(split)] = checkpoint.pop(key)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Time: {time.time() - t0:.02f} seconds.")

        model.eval()

    small_lm_head = model.lm_head

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    while True: 

        prompt = input("Type prompt (or 'exit'): ")

        if prompt == 'exit':
            break

        prompt = prompt.strip()

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)

        repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                            sample_until_period=sample_until_period, 
                            addl_token_limit=addl_token_limit,
                            verbose=True)
    return




if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)