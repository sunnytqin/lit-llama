import random
import torch

from data import (
    get_answer,
    SyntheticRepetitionTokenizer,
)
from model import (
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


while True:
    no_questions = int(input(f"Enter the number of questions in this prompt (1 to {questions_per_sample}): "))
    if(not 1 <= no_questions <= questions_per_sample):
        print("Input out of range. Try again.")
        continue

    prompt = ""
    valid = True
    gt = ''
    for i in range(no_questions):
        e_or_a = input(f"{i + 1}. Is the question epistemic or aleatoric? (e/a): ")
        if(not e_or_a in ["e", "a"]):
            print("Invalid input. Try again.")
            valid = False
            break

        first_bit = {"e": 0, "a": 1}[e_or_a]
        question_max = 2 ** (question_length - 1) - 1
        question = int(input(f"{i + 1}. Enter the question (0 to {question_max}): "))
        if(not 0 <= question <= question_max):
            print("Invalid input. Try again.")
            valid = False
            break

        question_bits = bin(question)[2:].zfill((question_length - 1))

        if(e_or_a == 'e'):
            answer = get_answer(question)
        else:
            answer = random.randint(0, 1)

        prompt += f"{first_bit}{question_bits}"

        if(i != no_questions - 1):
            prompt += f"{answer}<eos>"
        else:
            gt = answer

    if(not valid):
        continue

    tokenized_prompt = tokenizer.encode(prompt, parse_special_tokens=True)
    tokenized_prompt = torch.tensor(tokenized_prompt).unsqueeze(0).to("cuda")
    output, logits = model.generate(
        tokenized_prompt,
        max_new_tokens=1,
        return_logits=True,
    )

    output = tokenizer.decode(output[0].tolist())

    model_answer = output[-1]
    if(e_or_a == 'e'):
        print(f"Model answer: {model_answer}")
        print(f"Ground truth: {gt}")

    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits = logits[0, 0, list(range(tokenizer.vocab_size))]
    print(logits.shape)
    print(f"Logits: {list(zip(tokenizer.combined_tokens, logits.tolist()))}")


    print(output)
