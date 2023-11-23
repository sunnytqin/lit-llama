import json
import os
import pickle
import sys

# sys.path.append("/n/home12/gahdritz/projects/hallucinations_uncertainty")

import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import pipeLLaMA, LLaMAConfig
from wikidata.templates import TEMPLATES
from wikidata.train_head_utils import (
    load_llama,
)

DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')
MAX_SENTENCES_PER_TYPE = 5000
FW = True

model_size = "7B"
checkpoint_path = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth"
tokenizer_path = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model"
data_pairs_path = "/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/filtered_data_plaintext_sample.json"

model, tokenizer = load_llama(
    model_size, checkpoint_path, tokenizer_path, DTYPE, quantize=None, return_tokenizer_as_fn=False
)

with open(data_pairs_path, "r") as f:
    data_pairs = json.load(f)

# A surprise tool that will help us later
# (a lone period produces a different token ID for some idiotic reason)
period_id = tokenizer.encode("Period.", bos=False, eos=False, device=DEVICE)[-1].item()
addl_token_limit = 10

# LLaMA loves to output numbered lists of outputs
one_id = tokenizer.encode("1\".", bos=False, eos=False, device=DEVICE)[-2].item()

#data = {}
for property in data_pairs:
    data = {}
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
        for j, value in enumerate(values):
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

                encoded_sentence = tokenizer.encode(sentence, bos=True, eos=False, device=DEVICE)
                encoded_sentence = encoded_sentence.unsqueeze(0) 
                prompt_len = encoded_sentence.shape[-1]

                # embeds = model._forward(encoded_sentence).detach().cpu()

                ##### VISUAL INSPECTION #####
                addl_tokens = 0
                while True:
                    logits = model(encoded_sentence).detach().cpu()
                    best_token = torch.argmax(logits, dim=-1)[:, -1].to(DEVICE)
                    encoded_sentence = torch.cat(
                        [
                            encoded_sentence,
                            best_token[:, None],
                        ],
                        dim=-1
                    )

                    if(
                        (best_token == period_id and encoded_sentence[..., -2] != one_id) or
                        best_token == tokenizer.eos_id
                    ):
                        break

                    addl_tokens += 1
                    if(addl_tokens >= addl_token_limit):
                        break
               
                generated_tokens = encoded_sentence[0, prompt_len:]
                decoded_sentence = tokenizer.decode(generated_tokens)
                print(f"Prompt: {sentence}")
                print(f"Prediction: {decoded_sentence}")
                print(f"Key: {key}")
                print(f"Value: {value}")
                is_correct = value in decoded_sentence

                #print(is_correct)
                ##### END VISUAL INSPECTION #####

                count += 1
                if(count % 100 == 0):
                    print(f"Count: {count}")

                # if(is_correct):
                #     label = f"{property}_{i}_{j}_{k}_key_{key}_value_{value}"
                #     data[label] = embeds

                if(count >= MAX_SENTENCES_PER_TYPE):
                    break
            
            if(count >= MAX_SENTENCES_PER_TYPE):
                break
        
        if(count >= MAX_SENTENCES_PER_TYPE):
            break

    if(FW):
        filename = "logits_7B_fw.pickle"
    else:
        filename = "logits_7B_bw.pickle"

    filename = f"{property}_correct_{filename}"
    filename = os.path.join("logits/spares/", filename)

    with open(filename, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

# if(FW):
#     filename = "logits_7B_fw.pickle"
# else:
#     filename = "logits_7B_bw.pickle"

# with open(filename, "wb") as fp:
#     pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)



