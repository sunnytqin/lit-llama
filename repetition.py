import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA, LLaMA
from lit_llama.utils import EmptyInitOnDevice
from train_head_utils import PrecomputedShardLoader
import json
import pickle
from pathlib import Path
from tqdm import tqdm

DEVICE= "cuda"
# DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
DTYPE = torch.float32

def log(file, *content, **kwargs):
    if file is not None:
        with open(file, "a") as f: print(content, **kwargs, file=file)
    else: 
        print(*content, **kwargs)


def main(
    repetition_filter: str,
    shard_count: str, 
    prompts_json_path: str,
    model_size: str,
    experiment_name: str = None,
    k: int = 10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,
    sample_until_period: bool = True,
    addl_token_limit: int = 100,
    example_path: str = None, 
):
    """
    Args:
        repetition_filter: 
        shard_count: 
        prompts_json_path: The small LM checkpoint path.
        model_size: 
        k: 
        tokenizer_path: 
        checkpoint_path: 
        sample_until_period: 
        addl_token_limit: 
        example_path_print:  
    """

    if(checkpoint_path is None):
        checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth")

    if(tokenizer_path is None):
        tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()


    # Initialize the model
    # with EmptyInitOnDevice(
    #     device=DEVICE, dtype=DTYPE, quantization_mode=None,
    # ):
    #     print("Loading model... ", end='')
    #     t0 = time.time()
    #     model = pipeLLaMA.from_name(model_size)
    #     partition_schedule = model.partition_schedule
    #     checkpoint = torch.load(checkpoint_path)
    #     for key in list(checkpoint.keys()):
    #         if 'transformer.h' in key:
    #             split = key.split('.')
    #             split[2] = partition_schedule[int(split[2])]
    #             checkpoint[".".join(split)] = checkpoint.pop(key)
    #     model.load_state_dict(checkpoint, strict=True)
    #     print(f"Time: {time.time() - t0:.02f} seconds.")

    with EmptyInitOnDevice(
        device=DEVICE, dtype=DTYPE, quantization_mode=None,
    ):
        print("Loading model ...", end='', file=sys.stderr)
        t0 = time.time()
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    # load prompt indicator 
    shard_count = int(shard_count)
    prompt_filter_fp = os.path.join(repetition_filter, "filter", f"filter_{shard_count}.pickle")
    # prompt_loader = PrecomputedShardLoader([prompt_filter_fp])
    with open(prompt_filter_fp, "rb") as fp:
        prompt_loader = pickle.load(fp)
            
    large_entropy_dict_path = f"{repetition_filter}/large_entropy/large_entropy_{shard_count}.pickle"
    with open(large_entropy_dict_path, "rb") as fp:
        large_entropy_dict = pickle.load(fp)
    small_entropy_dict_path = f"{repetition_filter}/small_entropy/small_entropy_{shard_count}.pickle"
    with open(small_entropy_dict_path, "rb") as fp:
        small_entropy_dict = pickle.load(fp)

    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)
    
    encoded_prompts = []
    prompt_type = []
    promot_indicator_sum = 0
    large_entropy_all = []
    for i, (prompt_key, promot_indicator) in enumerate(prompt_loader.items()):
        if promot_indicator.sum()  == 0: continue
        print(prompt_key, promot_indicator.sum())
        encoded_prompt = tokenizer.encode(prompts[prompt_key], bos=True, eos=False, device=DEVICE)
        large_entropy_array = large_entropy_dict[prompt_key]
        small_entropy_array = small_entropy_dict[prompt_key]
        assert promot_indicator.shape[-1] == encoded_prompt.shape[0]
        assert promot_indicator.shape[-1] == large_entropy_array.shape[0] == small_entropy_array.shape[0]
        promot_indicator_sum += promot_indicator.sum()

        for eligible_index in torch.argwhere(promot_indicator):
            if eligible_index < 500:
                encoded_prompts.append(encoded_prompt[None, :eligible_index+1])
                large_entropy = large_entropy_array[eligible_index].double()
                large_entropy_all.append(large_entropy)
                prompt_type.append(large_entropy>0.2)
                # small_entropy = small_entropy_array[eligible_index].double()
    prompt_type = torch.LongTensor(prompt_type)
    log(example_path, prompt_type)

    print(f"{len(prompt_type)} encoded prompts , w/ {prompt_type.sum()} low_e_high_a examples.", file=sys.stderr) 

    if experiment_name is not None:
        save_dir = os.path.join(repetition_filter, experiment_name)
        os.makedirs(save_dir, exist_ok=True)

    else:
        save_dir = repetition_filter
    
    # large model small entropy example "high_e_low_a"
    new_embed_all = []
    orig_embed_all = []
    for i, encoded_prompt in enumerate(encoded_prompts):
        if i != 0 and i % 10 == 0: 
            print(f"{i}, {time.time() - t0:.02f} seconds.", file=sys.stderr)
            t0 = time.time()
        sys.stdout.flush()
        torch.cuda.empty_cache() 
        orig_embed, repetition_embeds = repetition_experiment(model, encoded_prompt, tokenizer, k,
                                                              sample_until_period=sample_until_period,
                                                              addl_token_limit=addl_token_limit,
                                                              example_path=example_path)
        if ~prompt_type.bool()[i]:
            log(example_path, "high_e_low_a example: ")
        else:
            log(example_path, "low_e_high_a example: ")
        orig_embed_all.append(orig_embed)
        new_embed_all.append(repetition_embeds)

    new_embed_all = torch.concatenate(new_embed_all)
    orig_embed = torch.concatenate(orig_embed_all)
    torch.save({"new_embed": new_embed_all, 
                "original_embed": orig_embed_all, 
                "large_entropy": large_entropy_all, 
                "prompt_type": prompt_type}, 
                f'{save_dir}/repetition_{shard_count}.pt')
    
    return


def repetition_experiment(model, encoded_prompt, tokenizer, k, 
                          sample_until_period=True,
                          addl_token_limit=100,
                          example_path=None, 
                          ):
    
    len_prompt = encoded_prompt.shape[-1]
    log(example_path, f"\nPrompt: \n {tokenizer.decode(encoded_prompt)}")
    # Run the model
    with torch.no_grad():
        orig_embed = model._forward(encoded_prompt).detach()
        generated = model.lm_head(orig_embed).detach()
    generated = torch.softmax(generated, dim=-1).detach().cpu()
    orig_embed = orig_embed[0, -1, :].detach().cpu()

    # Top k tokens
    log(example_path, "\nTop K Token: \n")
    top_k = torch.topk(generated, k, dim=-1).indices[0, -1, :].to(DEVICE)

    for t in torch.unbind(top_k):
        log(example_path, f"{tokenizer.decode(t)}: {generated[0, -1, t]:.3f}", end=" ")

    log(example_path, "\n \nTop K Repetition: \n")

    # A surprise tool that will help us later
    # (a lone period produces a different token ID for some idiotic reason)
    period_id = tokenizer.encode("Period.", bos=False, eos=False, device=DEVICE)[-1].item()

    repetition_embeds = []
    for t in torch.unbind(top_k):
        prompt_with_candidate = torch.cat(
            [
                encoded_prompt,
                t[None, None],
            ],
            dim=-1
        )

        if(sample_until_period):
            addl_tokens = 0
            while True:
                repetition_logits = model.forward(prompt_with_candidate).detach().cpu()
                best_token = torch.argmax(repetition_logits, dim=-1)[:, -1].to(DEVICE)
                prompt_with_candidate = torch.cat(
                    [
                        prompt_with_candidate,
                        best_token[:, None],
                    ],
                    dim=-1
                )

                if(best_token == period_id or best_token == tokenizer.eos_id):
                    break

                addl_tokens += 1
                if(addl_tokens >= addl_token_limit):
                    break

        log(example_path, "[prompt]", tokenizer.decode(prompt_with_candidate[0, len_prompt:]), end="<EOS> [prompt]")

        repetition_prompt = torch.cat(
            [
                prompt_with_candidate,
                torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
                encoded_prompt,
            ],
            dim=-1
        )

        with torch.no_grad():
            repetition_embed = model._forward(repetition_prompt)[:, -1, :].detach().cpu()
            repetition_embeds.append(repetition_embed)
            if True:
                repetition_logits = model.lm_head(repetition_embed.to(DEVICE))[0, :].detach()
                repetition_logits = torch.softmax(repetition_logits, dim=-1)
                repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                for d, p in zip(decoded, prob):
                    log(example_path, f"{d}: {p:.3f}", end=" ")
                log(example_path, "\n")    
    return orig_embed, torch.concatenate(repetition_embeds)[None, :]


def compute_entropy(logits):
    logits_softmax = torch.softmax(logits, dim=-1)
    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = torch.sum(-1 * logits_softmax * logs, dim=-1)
    return entropy



if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)