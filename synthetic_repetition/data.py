import random

import torch

"""
Generates a synthetic dataset for the purposes of testing the repetition 
approach.

Entries are of the form:

    <Q><A>

where:

<Q> is a binary string of a fixed length. The first bit of the determines 
whether the question is "epistemic" (0) vs "aleatoric" (1).
<A> is a single bit representing the answer. For aleatoric questions, it is 
sampled uniformly at random every time the question is sampled. For epistemic 
questions, it is determined by:

    g = random.Random(<Q>)
    <A> = g.randint(0, 1)
"""

class SyntheticRepetitionTokenizer:
    def __init__(self):
        self.tokens = [
            '0', '1',
        ]
        self.special_tokens = [
            '<pad>', '<eos>',
        ]
        self.token_set = set(self.tokens)
        self.special_token_set = set(self.special_tokens)

        assert(len(self.tokens) == len(self.token_set))
        assert(len(self.special_tokens) == len(self.special_token_set))

        combined_tokens = self.tokens + self.special_tokens
        self.id_to_token = {k:v for k, v in enumerate(combined_tokens)}
        self.token_to_id = {v:k for k, v in self.id_to_token.items()}
        self.eos_token_id = self.token_to_id['<eos>']
        self.vocab_size = len(combined_tokens)

    def encode(self, q):
        assert(set(q).issubset(self.token_set))
        return list(map(self.token_to_id.get, q))
    
    def decode(self, q):
        return ''.join(map(self.id_to_token.get, q))


def generate_synthetic_repetition_dataset(
    question_length,
    epistemic_prob=0.5,
    questions_per_sample=1,
    seed=42,
):
    random_gen = random.Random(seed)

    assert(question_length > 0)

    sample = []
    while True:
        # Determine if the question is epistemic or aleatoric
        first_bit = 0 if random_gen.random() < epistemic_prob else 1

        # Generate the rest of the question
        remaining_bits = ''
        if(question_length > 1):
            q = random_gen.randint(0, 2 ** (question_length - 1) - 1)
            remaining_bits = bin(q)[2:].zfill((question_length - 1))
            
        q_str = f"{first_bit}{remaining_bits}"

        # Generate the answer
        if first_bit == 0:
            # Epistemic
            answer_gen = random.Random(q)
            a = answer_gen.randint(0, 1)
        else:
            # Aleatoric
            a = random_gen.randint(0, 1)

        sample.append((q_str, str(a)))

        if(len(sample) == questions_per_sample):
            yield list(zip(*sample))

            sample = []


class SyntheticRepetitionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        question_length,
        epistemic_prob=0.5,
        questions_per_sample=1,
        seed=42,
    ):
        self.question_length = question_length
        self.epistemic_prob = epistemic_prob
        self.questions_per_sample = questions_per_sample
        self.seed = seed

    def __len__(self):
        return float('inf')

    def __iter__(self):
        return generate_synthetic_repetition_dataset(
            self.question_length,
            self.epistemic_prob,
            self.questions_per_sample,
            self.seed,
        )


if __name__ == "__main__":
    counts = {}
    for i, (q, a,) in enumerate(generate_synthetic_repetition_dataset(10)):
        q, = q
        a, = a

        if q[0] == "0":
            counts[a] = counts.get(a, 0) + 1

        if(i > 10000):
            break

    print(counts)
    