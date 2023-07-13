import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import copy 
import numpy as np

import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, jsd
from lit_llama.model import pipeLLaMA, LLaMAConfig


DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty
    probs_list = torch.empty(max_new_tokens, 32_000, dtype=DTYPE, device=idx.device)
    top_k_indices = torch.empty(max_new_tokens, 5, dtype=torch.int)
    top_k_probs = torch.empty(max_new_tokens, 5)
    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        if(max_new_tokens == 1):
            print(f"idx_cond: {idx_cond.shape}")

        # forward
        logits = model(idx_cond)

        if(max_new_tokens == 1):
            print(f"logits: {logits.shape}")
            print(f"logits: {logits[0, 0, :100]}")
        logits = logits[:, -1] / temperature

        # # optionally crop the logits to only the top k options
        # if top_k is not None:
        #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #     logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1)
        # idx_next = torch.multinomial(probs, num_samples=1)
        top_k = torch.topk(probs, 5, dim=-1)
        top_k_probs[t-T, :] = top_k[0]
        top_k_indices[t - T, :] = top_k[1]

        # concatenate the new column
        idx[:, t:] = idx_next
        probs_list[t - T, :] = probs.squeeze()

    return idx, probs_list, (top_k_probs, top_k_indices)


def main(
    prompt: str = "Hello, my name is",
    *,
    model_size: str = "7B",
    num_samples: int = 1,
    max_new_tokens: int = 20,
    top_k: int = 200,
    temperature: float = 1.0, # fix to lowest temperature
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not checkpoint_path:
        checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/state_dict.pth')
    if not tokenizer_path:
        tokenizer_path = Path('/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model')
    
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    large_model_size = "30B"
    large_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/checkpoints/lit-llama/{large_model_size}/state_dict.pth" 
    
    device_large = torch.device('cuda:0')
    device_small = torch.device('cuda:2')
  
    print("Loading large model ...", file=sys.stderr, end='')
    with EmptyInitOnDevice(
        device=device_large, dtype=DTYPE, quantization_mode=quantize
    ):
        t0 = time.time()
        large_model = pipeLLaMA.from_name(large_model_size)
        partition_schedule = large_model.partition_schedule
        checkpoint = torch.load(large_checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        large_model.load_state_dict(checkpoint)
    print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    large_model.eval()

    with EmptyInitOnDevice(
        device=device_small, dtype=DTYPE, quantization_mode=quantize
    ):
        print("Loading small model ...", file=sys.stderr, end='')
        t0 = time.time()
        small_model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        small_model.load_state_dict(checkpoint)
        print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    small_model.eval()

    tokenizer = Tokenizer(tokenizer_path)

    small_sentence_probs_meta = []
    large_sentence_probs_meta = []
    small_sentence_pieces_meta = []
    large_sentence_pieces_meta = []
    JSD_meta = []
    prompts_meta = []

    while True: 

        prompt = input("Type prompt (or 'exit'): ")

        if prompt == 'exit':
            break

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=device_small)
        len_prompt = len(encoded_prompt)
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension

        # L.seed_everything(1234)

        t0 = time.perf_counter()
        y, probs, top_k = generate(
            small_model,
            encoded_prompt,
            max_new_tokens,
            small_model.config.block_size,  # type: ignore[union-attr,arg-type]
            # temperature=temperature,
            # top_k=top_k,
        )
        y = y[0]  # unpack batch dimension
        t = time.perf_counter() - t0

        # print(f"\n\nTime for inference: {t:.02f} sec total, {num_samples * max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
        # print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
        y_compare = [*encoded_prompt[0]]
        probs_compare = torch.empty(max_new_tokens, 32_000, dtype=torch.bfloat16, device=encoded_prompt.device)
        top_k_probs, top_k_indices = top_k
        top_k_indices_compare = torch.empty_like(top_k_indices, dtype=torch.int)
        top_k_probs_compare = torch.empty_like(top_k_probs)
        for t in range(len_prompt, len(y)):
            encoded_prompt_large = y[0: t][None, :].to(device_large)

            print(large_model.config.block_size)

            t0 = time.perf_counter()
            y_large, probs_large, top_k_large = generate(
                large_model,
                encoded_prompt_large,
                1, #max new token is limited to 1
                large_model.config.block_size,  # type: ignore[union-attr,arg-type]
                temperature=temperature,
                top_k=top_k,
            )

            y_large = y_large[0]  # unpack batch dimension
            y_compare.append(y_large[-1].detach().int().to(cuda2))
            probs_compare[t - len_prompt, :] = probs_large.squeeze()

            top_k_indices_compare[t - len_prompt, :] = top_k_large[1]
            top_k_probs_compare[t - len_prompt, :] = top_k_large[0]
        y_compare = torch.stack(y_compare, dim=0)

        t = time.perf_counter() - t0

        # compute JS distance
        distance = jsd(probs, probs_compare)
        print(f"distance: {distance.shape}")
        distance = distance.tolist()
        JSD_meta.append(distance)

        # topk prediction
        small_sentence_pieces_j = []
        large_sentence_pieces_j = []
        for j in range(5):
            small_sentence = tokenizer.decode(top_k_indices_compare[:, j])#.replace("\n","\\n")
            large_sentence = tokenizer.decode(top_k_indices[:, j])#.replace("\n","\\n")

            small_sentence_pieces_j.append([p.piece for p in small_sentence.pieces])
            large_sentence_pieces_j.append([p.piece for p in large_sentence.pieces])

        small_sentence_pieces_meta.append(small_sentence_pieces_j)
        large_sentence_pieces_meta.append(large_sentence_pieces_j)

        small_sentence_probs_meta.append(top_k_probs.numpy())
        large_sentence_probs_meta.append(top_k_probs_compare.numpy())

        # # top1 prediction
        # small_sentence = tokenizer.decode(y)#.replace("\n","\\n")
        # large_sentence = tokenizer.decode(y_compare)#.replace("\n","\\n")
        
        # for i, (sp, bp) in enumerate(zip(small_sentence.pieces[len_prompt:], large_sentence.pieces[len_prompt:])):
        #     if sp.id == bp.id:
        #         print(f"{sp.piece}({torch.max(probs[i]):.2f}), {bp.piece}({torch.max(probs_compare[i]):.2f})")
        #         # print('piece="{}" surface="{}" id={} begin={} end={}'.format(n.piece, n.surface, n.id, n.begin, n.end))

    

    np.savez("sample_output/user_input.npz",
             small_sentence_probs=np.array(small_sentence_probs_meta),
             large_sentence_probs=np.array(large_sentence_probs_meta),
             small_sentence_pieces=np.array(small_sentence_pieces_meta),
             large_sentence_pieces=np.array(large_sentence_pieces_meta),
             JSD=JSD_meta,
             prompts=prompts_meta
             )

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
