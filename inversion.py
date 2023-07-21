import sys
import time

from jsonargparse import CLI
import torch

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA
from lit_llama.utils import EmptyInitOnDevice


DEVICE="cuda"
DTYPE=torch.float32
MASK_PROMPT = (
    "In the following sentences, replace '[MASK]' with a short, relevant word or phrase. \n "
    "SENTENCE: The [MASK] is a large animal that lives in Africa \n "
    "ANSWER: elephant \n "
    "SENTENCE: [MASK] is the capital of Portugal \n "
    "ANSWER: Lisbon \n "
    "SENTENCE: [MASK] is a popular game show hosted by Pat Sajak \n "
    "ANSWER: Wheel of Fortune \n "
    "SENTENCE:"
)

def main(
    prompt: str,
    model_size: str,
    k: int = 10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,
    mask_begin: int = None,
    mask_end: int = None,
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

    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE)

    print(encoded_prompt)
    for i, t in enumerate(torch.unbind(encoded_prompt)):
        print(f"{i}: {tokenizer.decode(t)}")
    
    print("(end of prompt tokens)")

    # Add batch dimension
    encoded_prompt = encoded_prompt[None, :]

    # Generate
    generated = model.forward(encoded_prompt)
    generated = torch.softmax(generated, dim=-1)

    # Top k tokens
    topk = torch.topk(generated, k, dim=-1).indices[0, -1, :]

    for t in torch.unbind(topk):
        print(f"{tokenizer.decode(t)}: {generated[0, -1, t]}")

    print("(end of top k tokens)")

    # Masked prompt
    prefix = encoded_prompt[:, 1:mask_begin] # cut BOS
    suffix = encoded_prompt[:, mask_end + 1:]
    masked_prompt = torch.cat(
        [
            prefix,
            tokenizer.encode("[MASK]", bos=False, eos=False, device=DEVICE)[None, :],
            suffix,
        ],
        dim=-1
    )

    total_log_probs = []
    entropy_lists = []
    for t in torch.unbind(topk):
        masked_prompt_with_candidate = torch.cat(
            [
                masked_prompt,
                t[None, None],
            ],
            dim=-1
        )

        few_shot_prompt = torch.cat(
            [
                tokenizer.encode(MASK_PROMPT, bos=True, eos=False, device=DEVICE)[None, :],
                masked_prompt_with_candidate,
                tokenizer.encode("\n ANSWER:", bos=False, eos=False, device=DEVICE)[None, :],
                encoded_prompt[:, mask_begin:mask_end + 1],
            ],
            dim=-1
        )

        print(tokenizer.decode(few_shot_prompt))

        with torch.no_grad():
            logits = model.forward(few_shot_prompt)
            logits = torch.nn.functional.softmax(logits, dim=-1)

        entropies = torch.sum(-1 * logits * torch.log(logits), dim=-1)

        answer_logits = logits[:, -(mask_end + 1 - mask_begin) - 1: -1]
        answer_probs = torch.gather(
            answer_logits, 
            -1, 
            encoded_prompt[:, mask_begin:mask_end + 1, None].long()
        )

        answer_entropies = entropies[:, -(mask_end + 1 - mask_begin) - 1: -1]

        #score = sum([l / e for l, e in zip(answer_probs[0], answer_entropies[0])])
        #score = sum([l * e for l, e in zip(answer_probs[0], answer_entropies[0])])
        #score = sum([(l * torch.log(l) * -1)/ e for l, e in zip(answer_probs[0], answer_entropies[0])])
        # score *= generated[0, -1, t]
        
        #answer_probs = sum([l / torch.max(p) for l, p in zip(answer_probs[0], answer_logits[0])])
        log_answer_probs = torch.log(answer_probs)
        total_log_probs.append(log_answer_probs.sum())
        #total_log_probs.append(sum([l / e for l, e in zip(neg_log_answer_probs[0], answer_entropies[0])]))
        #total_log_probs.append(score)
        entropy_lists.append(answer_entropies[0])

    sorted_candidates = list(sorted(
        zip(topk, total_log_probs, entropy_lists),
        key=lambda t: t[1],
        reverse=True,
    ))

    print("Top candidates:")
    for t, p, el in sorted_candidates:
        print(f"{tokenizer.decode(t)} ({t}, {el}): {p}")

        


if __name__ == "__main__":
    CLI(main)
