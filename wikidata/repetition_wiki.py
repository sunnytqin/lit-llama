import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn

import json
import pickle
from pathlib import Path
from tqdm import tqdm
from train_head_utils import (
    load_llama,
    load_pythia,
    load_icml,
    MAX_LEN,
    load_lm_head,
)
from transformers import (
    AutoTokenizer,
)
from lit_llama import Tokenizer
DEVICE= "cuda"
# DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
DTYPE = torch.float32

def log(file, *content, **kwargs):
    if file is not None:
        with open(file, "a") as f: print(content, **kwargs, file=file)
    else: 
        print(*content, **kwargs)


def pythia_forward(model, embeddings=False, return_after_layer_n=-1):
    def fw(input_ids):
        outputs = model.gpt_neox(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[return_after_layer_n]

        if(embeddings):
            return hidden_states

        logits = model.embed_out(hidden_states)
        return logits

    return fw

def main(
    model_type: str,
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

    if model_type == 'llama':
        PATH_PREFIX = f"/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/llama_{model_size}"
    elif model_type == 'icml':
        PATH_PREFIX = f"/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/icml"


    if not checkpoint_path:
        if(model_type == "llama"):
            checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth')
        elif(model_type == "pythia"):
            checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/pythia/pythia-{model_size}/')
        elif(model_type == "icml"):
            checkpoint_path = Path(f'.')
        else:
            raise ValueError
        print(checkpoint_path)
    
    if not tokenizer_path:
        if(model_type == "llama" or model_type == "icml"):
            tokenizer_path = Path('/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model')
        elif(model_type == "pythia"):
            pass # Not necessary

    assert checkpoint_path.exists()
    if model_type == "llama": assert tokenizer_path.is_file()

    # Initialize the model and tokenizer
    if(model_type == "llama"):
        model, tokenizer = load_llama(
            model_size, checkpoint_path, tokenizer_path, DTYPE, None
        )
    elif(model_type == "pythia"):
        model, tokenizer = load_pythia(
            model_size, checkpoint_path, DTYPE
        )
    elif(model_type == "icml"):
        model, tokenizer = load_icml()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.eval()
    
    # load prompts 
    encoded_prompts = []
    prompt_type = []
    prompt_category = []
    for d in ['fw', 'bw']:
        prompts_json_path = os.path.join(PATH_PREFIX, f"wikidata_P17_{d}_2_3.json")
        with open(prompts_json_path, "r") as fp:
            prompts = json.load(fp)

        cnt = 0
        for i, (prompt_key, promot_val) in enumerate(prompts.items()):
            # if prompt_key.split('_')[0] != "P17":
            #     continue 
            # if i%2 != 0:
            #     continue 

            encoded_prompt = tokenizer(promot_val)
            
            encoded_prompts.append(encoded_prompt[None, :]) 
            prompt_category.append(prompt_key.split("_")[0])

            cnt += 1
         
        if d == 'fw':
            prompt_type.append(torch.zeros(cnt))
        else:
            prompt_type.append(torch.ones(cnt))
            
    prompt_type = torch.concatenate(prompt_type)
            
    print(f"{len(encoded_prompts)}, {len(prompt_type)} encoded prompts, bw: {prompt_type.sum()} ", file=sys.stderr) 
 
    # random_prompt = tokenizer("Mount Everest is located on the astronomical body Earth. "\
    #                           "Moon can be seen on the surface of Mare Nectaris. "\
    #                           "Hudson River is the body of water closest to The Statue of Liberty. "\
    #                           "London is located closest to the body of water River Thames. ")

    if experiment_name is not None:
        save_dir = os.path.join(PATH_PREFIX, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = PATH_PREFIX

    if model_type == "pythia":
        tokenizer= AutoTokenizer.from_pretrained(
            f"EleutherAI/pythia-{model_size}",
        )
    elif model_type == "llama" or model_type == "icml":
        tokenizer = Tokenizer(tokenizer_path)


    # for i in range(1, encoded_prompts[0].shape[1]):
    #     print(i, tokenizer.decode(encoded_prompts[0][0, i]), end="|")
    # print("  ")

    if model_type == "llama" or model_type == "pythia":
        small_lm_head = load_lm_head(
            checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=model_size
        )
    elif model_type == "icml":
        small_lm_head = model.lm_head
        small_lm_head = small_lm_head.eval()
    
    # repetition experiment
    new_embed_all = []
    orig_embed_all = []
    encoded_prompt_all = []
    t0 = time.time()
    for i, encoded_prompt in enumerate(encoded_prompts):
        if i != 0 and i % 100 == 0: 
            print(f"{i}, {time.time() - t0:.02f} seconds.", file=sys.stderr)
            t0 = time.time()
        sys.stdout.flush()
        torch.cuda.empty_cache() 
        orig_embed, entropy, repetition_embeds = repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k,
                                                              sample_until_period=sample_until_period,
                                                              addl_token_limit=addl_token_limit,
                                                              example_path=example_path,
                                                              random_prompt=None)
        # if ~prompt_type.bool()[i]:
        #     log(example_path, "high_e_low_a example: ")
        # else:
        #     log(example_path, "low_e_high_a example: ")
        
        orig_embed_all.append(orig_embed)
        new_embed_all.append(repetition_embeds)
        encoded_prompt_all.append(encoded_prompt.squeeze())

    new_embed_all = torch.concatenate(new_embed_all)
    orig_embed = torch.concatenate(orig_embed_all)
    torch.save({"new_embed": new_embed_all, 
                "original_embed": orig_embed_all, 
                "original_entropy": entropy, 
                "prompt_type": prompt_type,
                "prompt_category": prompt_category, 
                "encoded_prompt": encoded_prompt_all}, 
                f'{save_dir}/repetition.pt')
    
    return


def repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                          sample_until_period=True,
                          addl_token_limit=100,
                          example_path=None, 
                          random_prompt=None,
                          ):
    
    len_prompt = encoded_prompt.shape[-1]
    log(example_path, f"\nPrompt: \n {tokenizer.decode(encoded_prompt[0])}")
    # Run the model
    if model_type == "llama":
        with torch.no_grad():
            orig_embed = model._forward(encoded_prompt).detach()
            generated = model.lm_head(orig_embed).detach()
    elif model_type == "pythia":
        with torch.no_grad():
            orig_embed = pythia_forward(model, embeddings=True)(encoded_prompt)
            generated = small_lm_head(orig_embed)
    elif model_type == "icml":
        with torch.no_grad():
            output = model.forward(encoded_prompt, output_hidden_states=True)
            orig_embed = output.hidden_states[-1]
            generated = output[0].detach()

    entropy = compute_entropy(generated[0, -1, :])
    # if entropy < 2.0 or entropy > 3.0:
    #     print("entropy: ", generated.shape, entropy)
    generated = torch.softmax(generated, dim=-1).detach().cpu()

    orig_embed = orig_embed[0, -1, :].detach().cpu()        

    # Top k tokens
    log(example_path, "\nTop K Token: \n")
    top_k = torch.topk(generated, k, dim=-1).indices[0, -1, :].to(DEVICE)
    # top_k = torch.topk(generated, 50, dim=-1).indices[0, -1, 40:].to(DEVICE)

    for t in torch.unbind(top_k):
        log(example_path, f"{tokenizer.decode(t)}: {generated[0, -1, t]:.3f}", end=" ")

    log(example_path, "\n \nTop K Repetition: \n")

    # A surprise tool that will help us later
    # (a lone period produces a different token ID for some idiotic reason)
    if model_type == "llama" or model_type == "icml":
        period_id = tokenizer.encode('"United States of America".', 
                                     bos=False, eos=False, device=DEVICE)[-1].item()
        eos_id = tokenizer.eos_id
    elif model_type == "pythia":
        period_id = tokenizer.encode("Period.")[-1]
        eos_id = tokenizer.encode("<|endoftext|>")[0]
    
    repetition_embeds = []
    for t in torch.unbind(top_k):
        prompt_with_candidate = torch.cat(
            [
                encoded_prompt,
                t[None, None],
            ],
            dim=-1
        )
        # prompt_with_candidate = tokenizer.encode('Swizterland uses the currency Swiss franc', device=DEVICE)[None, :]

        if(sample_until_period):
            addl_tokens = 0
            while True:
                if model_type == "llama":
                    with torch.no_grad():
                        repetition_logits = model.forward(prompt_with_candidate).detach().cpu()
                elif model_type == "pythia":
                    with torch.no_grad():
                        repetition_logits = pythia_forward(model)(prompt_with_candidate).detach().cpu()

                # repetition_logits = model.forward(prompt_with_candidate).detach().cpu()
                best_token = torch.argmax(repetition_logits, dim=-1)[:, -1].to(DEVICE)
                prompt_with_candidate = torch.cat(
                    [
                        prompt_with_candidate,
                        best_token[:, None],
                    ],
                    dim=-1
                )

                if(best_token == period_id or best_token == eos_id):
                    break

                addl_tokens += 1
                if(addl_tokens >= addl_token_limit):
                    break

        log(example_path, "[prompt]", tokenizer.decode(prompt_with_candidate[0, len_prompt:]), end="<EOS> [prompt]")

        if model_type == 'pythia':
            repetition_prompt = torch.cat(
                    [
                        torch.tensor(eos_id, device=DEVICE)[None, None],
                        prompt_with_candidate,
                        torch.tensor(period_id, device=DEVICE)[None, None],
                        torch.tensor(eos_id, device=DEVICE)[None, None],
                        encoded_prompt,
                    ],
                    dim=-1
                )
        elif model_type == 'llama' or model_type == "icml":
            if random_prompt is None:
                repetition_prompt = torch.cat(
                        [
                            # torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
                            prompt_with_candidate,
                            # torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
                            # torch.tensor(tokenizer.bos_id, device=DEVICE)[None, None],
                            encoded_prompt[:, 1:], 
                        ],
                        dim=-1
                    )
            # else:
            #     repetition_prompt = torch.cat(
            #             [
            #                 prompt_with_candidate,
            #                 # torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
            #                 prompt_with_candidate,
            #                 torch.tensor(tokenizer.bos_id, device=DEVICE)[None, None],
            #                 encoded_prompt,
            #             ],
            #             dim=-1
            #         )

        if model_type == 'llama':
            with torch.no_grad():
                repetition_embed = model._forward(repetition_prompt)[:, -1, :].detach().cpu()
                repetition_embeds.append(repetition_embed)
                if False:
                    repetition_logits = model.lm_head(repetition_embed.to(DEVICE))[0, :].detach()
                    repetition_logits = torch.softmax(repetition_logits, dim=-1)
                    repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                    decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                    prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                    for d, p in zip(decoded, prob):
                        log(example_path, f"{d}: {p:.3f}", end=" ")
                    log(example_path, "\n")    
        if model_type == 'icml':
            with torch.no_grad():
                output = model.forward(repetition_prompt, output_hidden_states=True)
                repetition_embed = output.hidden_states[-1].detach()
                repetition_embeds.append(repetition_embed[:, -1, :])
                if True:
                    repetition_logits = output[0][0, -1, :].detach()
                    repetition_logits = torch.softmax(repetition_logits, dim=-1)
                    repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                    decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                    prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                    for d, p in zip(decoded, prob):
                        log(example_path, f"{d}: {p:.3f}", end=" ")
                    log(example_path, "\n")    
        elif model_type == "pythia":
            with torch.no_grad():
                repetition_embed = pythia_forward(model, embeddings=True)(repetition_prompt)[:, -1, :].detach().cpu()
                repetition_embeds.append(repetition_embed)
                if False:
                    repetition_logits = model.embed_out(repetition_embed.to(DEVICE))[0, :].detach()
                    repetition_logits = torch.softmax(repetition_logits, dim=-1)
                    repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                    decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                    prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                    for d, p in zip(decoded, prob):
                        log(example_path, f"{d}: {p:.5f}", end=" ")
                    log(example_path, "\n")    

    return orig_embed, entropy, torch.concatenate(repetition_embeds)[None, :]


def compute_entropy(logits):
    logits_softmax = torch.nn.functional.softmax(logits, dim=-1)
    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = torch.sum(-1 * logits_softmax * logs, dim=-1)
    return entropy



if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)