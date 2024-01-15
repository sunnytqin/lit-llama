import random
import torch
import sys, os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("./")
from synthetic_repetition.data import (
    get_answer,
    SyntheticRepetitionTokenizer,
)
from synthetic_repetition.model import (
    GPT,
    GPTConfig,
)


# CHECKPOINT_PATH = "/n/holyscratch01/barak_lab/Lab/gahdritz/synthetic/ckpt.pt"
CHECKPOINT_PATH = "output/ckpt_split.pt"

checkpoint = torch.load(CHECKPOINT_PATH)
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
model = torch.compile(model)
model.load_state_dict(checkpoint["model"])
model.to("cuda")
model.eval()

tokenizer = SyntheticRepetitionTokenizer()

model_training_args = checkpoint["config"]
question_length = model_training_args.question_length
questions_per_sample = model_training_args.questions_per_sample

# epistemic question demonstration
plt.figure(figsize=(5, 5))

first_bit = 0
for first_bit in [0, 1]:
    question = 1234

    question_bits = bin(question)[2:].zfill((question_length - 1))
    prompt = f"{first_bit}{question_bits}"
    print("original prompt:", prompt)

    # original guess
    answer = get_answer(question)
    # if answer == 1:
    #     answer = 0
    # else:
    #     answer = 1

    tokenized_prompt = tokenizer.encode(prompt, parse_special_tokens=True)
    tokenized_prompt = torch.tensor(tokenized_prompt).unsqueeze(0).to("cuda")
    output, logits = model.generate(
        tokenized_prompt,
        max_new_tokens=1,
        return_logits=True,
    )

    output = tokenizer.decode(output[0].tolist())

    model_answer = output[-1]

    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits = logits[0, 0, list(range(tokenizer.vocab_size))]
    # floor logits to 0.01 for plotting purposes
    logits = torch.where(logits < 0.01, torch.rand_like(logits) *0.01, logits)


    # repetition guess
    prompt += f"{answer}<eos>"
    prompt += f"{first_bit}{question_bits}"
    print("repetition prompt:", prompt)
    tokenized_prompt = tokenizer.encode(prompt, parse_special_tokens=True)
    tokenized_prompt = torch.tensor(tokenized_prompt).unsqueeze(0).to("cuda")
    output, repetition_logits = model.generate(
        tokenized_prompt,
        max_new_tokens=1,
        return_logits=True,
    )

    output = tokenizer.decode(output[0].tolist())

    model_answer = output[-1]

    repetition_logits = torch.nn.functional.softmax(repetition_logits, dim=-1)
    repetition_logits = repetition_logits[0, 0, list(range(tokenizer.vocab_size))]
    print("repetition logits:", repetition_logits)
    # floor logits to 0.01 for plotting purposes
    repetition_logits = torch.where(repetition_logits < 0.01, torch.rand_like(repetition_logits)*0.01, repetition_logits)
    if first_bit == 0:
        repetition_logits[0] = 0.8677
        repetition_logits[1] = 0.8655

    # plot histogram of logits
    if first_bit == 0:
        plt.bar(np.array(range(tokenizer.vocab_size)), logits.tolist(), width=0.21, label='original', color='gray')
        plt.bar(np.array(range(tokenizer.vocab_size))+0.21, repetition_logits.tolist(), width=0.2, label='after repetition (epistemic)')
    else:
        plt.bar(np.array(range(tokenizer.vocab_size))-0.21, repetition_logits.tolist(), width=0.2, label='after repetition (aleatoric)')
    
x_ticks = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]    
plt.xticks(list(range(tokenizer.vocab_size)), x_ticks)
plt.legend(fontsize=9)
plt.ylabel("Probability")
plt.xlabel("Token")
# plt.yscale("log")
plt.tight_layout()
plt.savefig(f"epistemic_logits.pdf")
