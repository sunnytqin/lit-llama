import math
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
        self.combined_tokens = self.tokens + self.special_tokens
        self.combined_token_set = set(self.combined_tokens)

        assert(len(self.tokens) == len(self.token_set))
        assert(len(self.special_tokens) == len(self.special_token_set))
        
        # To avoid headaches later, make sure that no token is a prefix of
        # another token
        for t1 in self.combined_tokens:
            for t2 in self.combined_tokens:
                if(t1 != t2):
                    assert(not t1.startswith(t2))

        self.id_to_token = {k:v for k, v in enumerate(self.combined_tokens)}
        self.token_to_id = {v:k for k, v in self.id_to_token.items()}
        self.eos_token_id = self.token_to_id['<eos>']
        self.vocab_size = len(self.combined_tokens)

    def encode(self, q, parse_special_tokens=False):
        if(parse_special_tokens):
            token_set = self.combined_token_set
        else:
            token_set = self.token_set

        tokens = []
        token_in_progress = ""
        for c in q:
            token_in_progress += c
            if(token_in_progress in token_set):
                tokens.append(token_in_progress)
                token_in_progress = ""
        
        assert(token_in_progress == "")

        return list(map(self.token_to_id.get, tokens))
    
    def decode(self, q):
        return ''.join(map(self.id_to_token.get, q))


def generate_synthetic_repetition_dataset(
    question_length,
    epistemic_prob=0.5,
    questions_per_sample=1,
    force_collision_prob=0.,
    seed=42,
    val=False,
):
    random_gen = random.Random(seed)

    assert(question_length > 1)
    if val:
        train_q = [*range(2**(question_length - 1))]
    else:
        test_q = [random_gen.randint(0, 2 ** (question_length - 1) - 1) for _ in range(10000)]
        print("test q: ", test_q[0:10])
        train_q = []
        for i in range(2**(question_length - 1)):
            if i not in test_q: train_q.append(i)

    sample = []
    previous_qs = []
    while True:
        # Determine if the question is epistemic or aleatoric
        first_bit = 0 if random_gen.random() < epistemic_prob else 1

        # Generate the rest of the question
        remaining_bits = ''
        q = train_q[random_gen.randint(0, len(train_q)-1)]
        
        if(first_bit == 0): # also repeating aleatoric questions break the model
            if(random_gen.random() < force_collision_prob and len(previous_qs) > 0 ):
                q = previous_qs[random_gen.randint(0, len(previous_qs) - 1)]
            else:
                previous_qs.append(q)
            
        remaining_bits = bin(q)[2:].zfill((question_length - 1))
        q_str = f"{first_bit}{remaining_bits}"

        # Generate the answer
        if first_bit == 0:
            # Epistemic
            a = get_answer(q)
        else:
            # Aleatoric
            a = random_gen.randint(0, 1)

        sample.append((q_str, str(a)))

        if(len(sample) == questions_per_sample):
            yield list(zip(*sample))

            sample = []
            previous_qs = [] 

def test_dataset_hint(question_length, exclusion = [[3, 7, 13], [4, 6, 14]]):
    inclusion = []
    for i in range(question_length - 1):
        if (i not in exclusion[0]) and (i not in exclusion[1]): inclusion.append(i)
    print("inclusion: ", inclusion)

    sample_questions = []
    for _ in range(10):
        empty_template = torch.zeros(question_length - 1, dtype=torch.long)
        empty_template[exclusion[0]] = 0
        empty_template[exclusion[1]] = 1
        empty_template[inclusion] = torch.randint(0, 2, [len(inclusion)])
        bin_q = ''.join(str(x) for x in empty_template.tolist())
        bin_q = int('0b' + bin_q, 2)
        sample_questions.append(bin_q)

    return sample_questions

def generate_synthetic_repetition_dataset_heavytail(
    question_length,
    epistemic_prob=0.5,
    questions_per_sample=1,
    force_collision_prob=0., 
    seed=42,
    **kwargs, 
):
    random_gen = random.Random(seed)
    # uniform = torch.distributions.log_normal.LogNormal(loc=8, scale=0.12)
    heavy_tail = torch.distributions.half_cauchy.HalfCauchy(scale=5.)

    assert(question_length > 1)

    sample = []
    # epistemic_qs = []
    while True:
        # Determine if the question is epistemic or aleatoric
        first_bit = 0 if random_gen.random() < epistemic_prob else 1

        # Generate the rest of the question
        remaining_bits = ''

        if first_bit == 1:
            q = random_gen.randint(0, 2 ** (question_length - 1) - 1)
        else:  
            q = torch.clip(heavy_tail.sample().to(torch.long), max=2 ** (question_length - 1) - 1)
        
        # if(first_bit == 0):
        #     # Epistemic
        #     if(random_gen.random() < force_collision_prob and len(epistemic_qs) > 0):
        #         # Force a collision
        #         q = epistemic_qs[random_gen.randint(0, len(epistemic_qs) - 1)]
        #     else:
        #         epistemic_qs.append(q)

        remaining_bits = bin(q)[2:].zfill((question_length - 1))
        q_str = f"{first_bit}{remaining_bits}"

        # Generate the answer
        if first_bit == 0:
            # Epistemic
            a = get_answer(q)
        else:
            # Aleatoric
            a = random_gen.randint(0, 1)

        sample.append((q_str, str(a)))

        if(len(sample) == questions_per_sample):
            # e_questions = [q[0] for q in sample if q[0][0] == '0']
            
            # if(len(set(e_questions)) < len(e_questions)):
            yield list(zip(*sample))

            sample = []
            # epistemic_qs = []


def get_answer(epistemic_question):
    answer_gen = random.Random(epistemic_question)
    a = answer_gen.randint(0, 1)
    return a


class SyntheticRepetitionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        question_length,
        epistemic_prob=0.5,
        questions_per_sample=1,
        force_collision_prob=0.,
        seed=42,
        heavy_tail=False,
        val=False,
    ):
        self.question_length = question_length
        self.epistemic_prob = epistemic_prob
        self.questions_per_sample = questions_per_sample
        self.force_collision_prob = force_collision_prob
        self.seed = seed
        self.range = range
        self.val = val
        if heavy_tail:
            self.generator = generate_synthetic_repetition_dataset_heavytail
        else:
            self.generator = generate_synthetic_repetition_dataset

    def __len__(self):
        return float('inf')

    def __iter__(self):
        return self.generator(
            question_length=self.question_length,
            epistemic_prob=self.epistemic_prob,
            questions_per_sample=self.questions_per_sample,
            force_collision_prob=self.force_collision_prob,
            seed=self.seed,
            val = self.val,
        )


def compute_epistemic_collision_prob(
    question_length,
    epistemic_prob,
    questions_per_sample
):
    """
       Returns the probability that at least two epistemic questions in a 
       multi-question prompt are the same
    """
    # Clean up notation
    p = epistemic_prob
    n = questions_per_sample
    
    if(n == 1):
        return 0

    # Probability of getting exactly k epistemic questions in the prompt
    prob_k_e = lambda k: (p ** k) * ((1 - p) ** (n - k)) * math.comb(n, k)

    # Probability that at least two of the epistemic questions are the same
    possible_questions = 2 ** (question_length - 1) # not counting e/a bit
    prob_collision = 0
    for k in range(2, n + 1):
        prob_all_different = 1
        for j in range(k):
            prob_all_different *= (possible_questions - j) / possible_questions
        
        prob_collision += prob_k_e(k) * (1 - prob_all_different)

    return prob_collision


if __name__ == "__main__":
    ql = 19
    ep = 0.5
    qps = 15
    fcp = 0.8

    print(f"Predicted: {compute_epistemic_collision_prob(ql, ep, qps)}")
    print("test set samples: ", test_dataset_hint(ql))

    gen = generate_synthetic_repetition_dataset(ql, ep, qps, fcp, val=False)
    e_collision_count = 0
    a_collision_count = 0
    total_examples = 1e4
    for _ in range(int(total_examples)):
        questions, answers = next(gen)
        
        # Keep the empirical ones
        e_questions = [q for q in questions if q[0] == '0']
        
        if(len(set(e_questions)) < len(e_questions)):
            e_collision_count += 1

        # Keep the empirical ones
        a_questions = [q for q in questions if q[0] == '1']
        
        if(len(set(a_questions)) < len(a_questions)):
            a_collision_count += 1
            

    print(f"Empirical: {e_collision_count / total_examples}, {a_collision_count / total_examples}")