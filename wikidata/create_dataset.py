import os, sys
import json
import numpy as np
from pathlib import Path

import torch

from wikidata.templates import TEMPLATES
from train_head_utils import (
    load_llama,
    load_pythia,
    load_icml,
    MAX_LEN,
    load_lm_head,
)


def qualitfy_sentence(model_type, tokenizer, model, sentence):
    # if model_type == "llama":
    #     encoded_prompt = tokenizer(sentence)[None, :]
    # elif model_type == "pythia":
    encoded_prompt = tokenizer(sentence)[None, :]
    
    if model_type == "llama":
        with torch.no_grad():
            orig_embed = model._forward(encoded_prompt).detach()
            generated = model.lm_head(orig_embed).detach()
    # elif model_type == "pythia":
    #     with torch.no_grad():
    #         orig_embed = pythia_forward(model, embeddings=True)(encoded_prompt)
    #         generated = small_lm_head(orig_embed)
    elif model_type == "icml":
        with torch.no_grad():
            output = model.forward(encoded_prompt, output_hidden_states=True)
            # orig_embed = output.hidden_states.detach()
            generated = output[0].detach()
    entropy = compute_entropy(generated[0, -1, :])
    
    return entropy
        
def main():
    model_type = "icml"
    model_size = "7B"
    checkpoint_path = None
    tokenizer_path = None

    DTYPE = torch.bfloat16
    MAX_SENTENCES_PER_TYPE = 1000
    FW = False
    PATH_PREFIX = "/n/holyscratch01/barak_lab/Lab/gahdritz/wikidata/"
    # MY_PREFIX = f"/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/llama_{model_size}/"
    MY_PREFIX = f"/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/icml/"

    if not checkpoint_path:
        if(model_type == "llama"):
            checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth')
        elif(model_type == "pythia"):
            checkpoint_path = Path(f'/n/holystore01/LABS/barak_lab/Everyone/pythia/pythia-{model_size}/')
        elif(model_type == "icml"):
            checkpoint_path = Path('.')
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

    data_pairs_path = os.path.join(PATH_PREFIX, "filtered_data_plaintext.json")
    with open(data_pairs_path, "r") as f:
        data_pairs = json.load(f)

    data_cnt = {}
    for property in data_pairs:
        if(not TEMPLATES[property]["type"] == "many_to_one"):
            continue
        cnt = 0
        for i, (key, values) in enumerate(data_pairs[property].items()):
            cnt+= len(values)
        data_cnt[property] = cnt
    print(data_cnt)

    data = {}
    for property in data_pairs:
        repeated_set = set()
        if property != "P17": continue # only want country data
       
        if(not TEMPLATES[property]["type"] == "many_to_one"):
            continue

        if(FW):
            prompt = TEMPLATES[property]["prompt_fw"]
        else:
            prompt = TEMPLATES[property]["prompt_bw"]

        few_shot_examples = TEMPLATES[property]["few_shot_examples"]
        if(FW):
            templates = TEMPLATES[property]["templates_fw"]
        else:
            templates = TEMPLATES[property]["templates_bw"]
        
        count = 0
        for i, (key, values) in enumerate(data_pairs[property].items()):
            if len(values) > 1: continue
            value = values[0]
            # for j, value in enumerate(values):
            if(FW):
                if key in repeated_set: continue
                repeated_set.add(key)
            else:
                if value in repeated_set: continue
                repeated_set.add(value)
            for k, template in enumerate(templates):
                if(FW):
                    build_sentence = lambda k: template.replace("<key>", k).split("<value>")

                    sentence = f"{prompt}\n"
                    for fs_key, fs_value in few_shot_examples:
                        fs_sentence_begin, fs_sentence_end = build_sentence(fs_key)
                        fs_sentence = f"{fs_sentence_begin}{fs_value}{fs_sentence_end}.\n"
                        sentence += fs_sentence

                    sentence += build_sentence(key)[0]
                else:
                    build_sentence = lambda v: template.replace("<value>", v).split("<key>")

                    sentence = f"{prompt}\n"
                    for fs_key, fs_value in few_shot_examples:
                        fs_sentence_begin, fs_sentence_end = build_sentence(fs_value)
                        fs_sentence = f"{fs_sentence_begin}{fs_key}{fs_sentence_end}.\n"
                        sentence += fs_sentence

                    sentence += build_sentence(value)[0]

                # Trailing whitespace can confuse the tokenizer (e.g. LLaMA adds a pre-numerical token)
                sentence = sentence.strip()
                entropy = qualitfy_sentence(model_type, tokenizer, model, sentence)
                if entropy > 2.0 and entropy < 3.0:
                    data[f"{property}_{i}_{k}"] = sentence
                    count += 1
        
                if(count >= MAX_SENTENCES_PER_TYPE):
                    break
            
            if(count >= MAX_SENTENCES_PER_TYPE):
                break
        
        print(f"Count: {count}")


    if(FW):
        filename = "wikidata_P17_fw_2_3.json"
    else:
        filename = "wikidata_P17_bw_2_3.json"

    filter_data_path = os.path.join(MY_PREFIX, filename)
    with open(filter_data_path, "w") as fp:
        json.dump(data, fp)
    
    return

def compute_entropy(logits):
    logits_softmax = torch.nn.functional.softmax(logits, dim=-1)
    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = torch.sum(-1 * logits_softmax * logs, dim=-1)
    return entropy


if __name__ == "__main__":

    main()