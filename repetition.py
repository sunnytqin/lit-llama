import sys
import time

from jsonargparse import CLI
import torch

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA
from lit_llama.utils import EmptyInitOnDevice


DEVICE="cuda"
DTYPE=torch.float32


def main(
    prompt: str,
    model_size: str,
    k: int = 10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,
    sample_until_period: bool = True,
    addl_token_limit: int = 100,
):
    if(checkpoint_path is None):
        checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth"

    if(tokenizer_path is None):
        tokenizer_path = "/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model"

    # Initialize the model
    with EmptyInitOnDevice(
        device=DEVICE, dtype=DTYPE, quantization_mode=None,
    ):
        t0 = time.time()
        model = pipeLLaMA.from_name(model_size)
        partition_schedule = model.partition_schedule
        checkpoint = torch.load(checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        model.load_state_dict(checkpoint)

    model.eval()

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    print(f"Prompt: {prompt}")

    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE)

    # Add batch dimension
    encoded_prompt = encoded_prompt[None, :]

    print(f"Encoded prompt: {encoded_prompt}")

    # Generate
    generated = model.forward(encoded_prompt)
    generated = torch.softmax(generated, dim=-1)

    # Top k tokens
    top_k = torch.topk(generated, k, dim=-1).indices[0, -1, :]

    for t in torch.unbind(top_k):
        print(f"{tokenizer.decode(t)} ({t}) ({bytes(tokenizer.decode(t), 'utf-8')}): {generated[0, -1, t]}")

    print("(end of top k tokens)")

    # A surprise tool that will help us later
    # (a lone period produces a different token ID for some idiotic reason)
    period_id = tokenizer.encode("Period.", bos=False, eos=False, device=DEVICE)[-1].item()

    tops_k = []
    probs = []
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
                repetition_logits = model.forward(prompt_with_candidate)
                best_token = torch.argmax(repetition_logits, dim=-1)[:, -1]
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

        print(tokenizer.decode(prompt_with_candidate[0, :]))

        repetition_prompt = torch.cat(
            [
                prompt_with_candidate,
                torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
                encoded_prompt,
            ],
            dim=-1
        )
        
        repetition_logits = model.forward(repetition_prompt)
        repetition_logits = torch.softmax(repetition_logits, dim=-1)
        repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices[0, -1, :]
        tops_k.append(repetition_top_k)
        probs.append([float(repetition_logits[0, -1, rt]) for rt in repetition_top_k])

    for t, repetition_top_k, prob in zip(top_k, tops_k, probs):
        decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
        print(f"{tokenizer.decode(t)}: {list(zip(decoded, prob))}")
        

if __name__ == "__main__":
    CLI(main)