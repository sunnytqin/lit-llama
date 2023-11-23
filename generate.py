import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA, LLaMA
from lit_llama.utils import EmptyInitOnDevice
from pathlib import Path

from repetition import repetition_experiment, compute_entropy

DEVICE= "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# DTYPE = torch.float32


def main(
    repetition: bool=False, 
    model_type: str = "llama", 
    model_size: str = "7B", 
    large_model_size: str = "30B",
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
    if repetition:
        generate_repetition(model_type=model_type, 
                            model_size=model_size,
                            k=k,
                            tokenizer_path=tokenizer_path,
                            checkpoint_path=checkpoint_path,
                            sample_until_period=sample_until_period,
                            addl_token_limit=addl_token_limit)
    else:
        generate_comparison(model_type=model_type, 
                            small_model_size=model_size,
                            large_model_size=large_model_size,
                            k=k,
                            tokenizer_path=tokenizer_path,
                            checkpoint_path=checkpoint_path,)

    return

def generate_repetition(
        model_type: str = "llama", 
        model_size: str = "7B",
        k: int = 10,
        tokenizer_path: str = None,
        checkpoint_path: str = None,
        sample_until_period: bool = True,
        addl_token_limit: int = 100,):

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
            quit()

        prompt = prompt.strip()

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)

        repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                            sample_until_period=sample_until_period, 
                            addl_token_limit=addl_token_limit,
                            verbose=True)
    return

def generate_comparison(model_type: str = "llama", 
    small_model_size: str = "7B",
    large_model_size: str = "30B",
    k=10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,):

    """
    Args:
        model_type: "llama", 
        small: [7B, 30B],
        k: top k token to display,
        tokenizer_path: llama tokenizer,
        checkpoint_path: model checkpoint default if None,
    """

    if(checkpoint_path is None):
        small_checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{small_model_size}/lit-llama.pth")
        large_checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{large_model_size}/lit-llama.pth")

    if(tokenizer_path is None):
        tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

    assert small_checkpoint_path.is_file()
    assert large_checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if small_model_size == "7B":
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model ...", end='', file=sys.stderr)
            t0 = time.time()
            small_model = LLaMA.from_name(small_model_size)
            checkpoint = torch.load(small_checkpoint_path)
            small_model.load_state_dict(checkpoint)
            print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    else: 
        # Initialize the model
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model... ", end='')
            t0 = time.time()
            small_model = pipeLLaMA.from_name(large_model_size)
            partition_schedule = small_model.partition_schedule
            checkpoint = torch.load(checkpoint_path)
            for key in list(checkpoint.keys()):
                if 'transformer.h' in key:
                    split = key.split('.')
                    split[2] = partition_schedule[int(split[2])]
                    checkpoint[".".join(split)] = checkpoint.pop(key)
            small_model.load_state_dict(checkpoint, strict=True)
            print(f"Time: {time.time() - t0:.02f} seconds.")

        small_model.eval()

    with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
        print("Loading model... ", end='')
        t0 = time.time()
        large_model = pipeLLaMA.from_name(large_model_size)
        partition_schedule = large_model.partition_schedule
        checkpoint = torch.load(large_checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        large_model.load_state_dict(checkpoint, strict=True)
        print(f"Time: {time.time() - t0:.02f} seconds.")

    large_model.eval()

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    while True: 

        prompt = input("Type prompt (or 'exit'): ")

        if prompt == 'exit':
            quit()

        prompt = prompt.strip()

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)
        print(f"\n{small_model_size} Model: ", end = " ")  
        small_logits = small_model(encoded_prompt.to(DEVICE))[0, -1, :].detach()
        entropy = compute_entropy(small_logits)
        small_logits = torch.softmax(small_logits, dim=-1)
        small_top_k = torch.topk(small_logits, k, dim=-1).indices
        decoded = [tokenizer.decode(rt) for rt in torch.unbind(small_top_k)]
        prob = [float(small_logits[rt]) for rt in small_top_k]
        for d, p in zip(decoded, prob):
            print(f"{d}: {p:.3f}", end=" ")
        print(f"(small entropy: {entropy:.3f})") 
        print(f"\n{large_model_size} Model: ", end = " ")   

        large_logits = large_model(encoded_prompt.to(DEVICE))[0, -1, :].detach()
        entropy = compute_entropy(large_logits)
        large_logits = torch.softmax(large_logits, dim=-1)
        large_top_k = torch.topk(large_logits, k, dim=-1).indices
        decoded = [tokenizer.decode(rt) for rt in torch.unbind(large_top_k)]
        prob = [float(large_logits[rt]) for rt in large_top_k]
        for d, p in zip(decoded, prob):
            print(f"{d}: {p:.3f}", end=" ")
        print(f"(large entropy: {entropy:.3f})") 
        print("\n")   

    
    return




if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)




if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)