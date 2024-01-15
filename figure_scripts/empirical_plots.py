import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("./")

from lit_llama import Tokenizer
from lit_llama.model import LLaMA
from lit_llama.utils import EmptyInitOnDevice
from pathlib import Path

from repetition import repetition_experiment


DEVICE= "cuda"
# DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
DTYPE = torch.float32
model_size = "7B"
model_type = "llama"
k = 10

checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth")
tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

assert checkpoint_path.is_file()
assert tokenizer_path.is_file()


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

small_lm_head = model.lm_head

# Initialize the tokenizer
tokenizer = Tokenizer(tokenizer_path)
# prompt = "36-42 Coney Street is a historic terrace in the city centre of"
prompt = "Bulgurluk is a village in the Genç District, Bingöl Province, Turkey. The village is"
# prompt = "Böbingen (Rems) station is a railway station in the municipality of Böbingen an der Rems, located in the"
prompt = prompt.strip()

encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)

original_embed, repetition_embeds = repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                        sample_until_period=False, 
                        addl_token_limit=100,
                        verbose=False)

original_logits = small_lm_head(original_embed.to(DEVICE)).detach()
original_probs = torch.softmax(original_logits, dim=-1)

topk = torch.topk(original_probs, k=k, dim=-1).values.detach().cpu()
topk_idx = torch.topk(original_probs, k=k, dim=-1).indices.cpu()
topk_token = []
print(topk_idx)
for rt in topk_idx:
    if tokenizer.decode(rt) == "":
        topk_token.append("<SPECIAL>")
    else:
        topk_token.append(f"'{tokenizer.decode(rt)}'")


new_logits = small_lm_head(repetition_embeds.to(DEVICE)).detach()
new_probs = torch.softmax(new_logits, dim=-1).detach().cpu()
repetition_probs = new_probs[0, range(k), topk_idx]

# plot the original probabilities as bars
plt.figure(figsize=(5, 5))
plt.bar(np.array(range(k))-0.15, topk, width=0.3, color='gray', label="original")
plt.bar(np.array(range(k))+0.15, repetition_probs, width=0.3, label="after repetition", color='C1')
plt.xticks(range(k), topk_token, rotation=90)
# plt.xlabel("Token")
plt.ylabel("Probability")
plt.legend(loc='upper center')
plt.title("Repetition for aleatoric example")
plt.savefig("aleatoric_logits.pdf", bbox_inches='tight')

