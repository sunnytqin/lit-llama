import json
import pickle

import torch
from transformers import AutoTokenizer

from train_head_utils import (
    discretize,
    DistancePredictionHead,
    load_pythia_model,
    load_lm_head,
    PrecomputedShardLoader,
    VOCAB_SIZES,
)

torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda:0")
DTYPE = torch.float32
MAX_LENGTH = 2048

classifier_checkpoint = "/n/holyscratch01/barak_lab/Lab/gahdritz/trained_heads/pythia-1.4b_pythia-12b_separated_classes_pile_code_eval//state_dict.pth"
model_type = "pythia"
small_model_size = "1.4b"
large_model_size = "12b"
no_bins = 2
min_bin = 0
max_bin = 1
small_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-{small_model_size}"
large_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-{large_model_size}"
precomputed_small_emb_dir_val = f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-{small_model_size}_val_val_code"
precomputed_large_emb_dir_val = f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-{large_model_size}_val_val_code"
val_dataset_filter_path = f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pile_pythia-{small_model_size}_pythia-{large_model_size}_val_val_code_3_max/filter.pickle"
raw_data_json = f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/pile_val_val.json"

tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/pythia-{small_model_size}",
)

small_model = load_pythia_model(small_checkpoint_path, model_size=small_model_size, dtype=DTYPE)
small_model = small_model.to(DEVICE)
small_model.eval()
small_lm_head = small_model.embed_out
large_lm_head = load_lm_head(large_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=large_model_size)

# Load the model
shared_head_params = {
    "no_bins": 2,
    "hidden_dim": 2048,
    "no_hidden_layers": 5,
    "dropout": 0.1,
    "activation": "relu",
}
distance_prediction_head = DistancePredictionHead(
    input_dim=small_lm_head.weight.shape[1],
    **shared_head_params,
)

distance_prediction_head.to(DEVICE)

# Umm uhh
#distance_prediction_head = torch.compile(distance_prediction_head)

checkpoint = torch.load(classifier_checkpoint, map_location=DEVICE)
distance_prediction_head.load_state_dict(checkpoint)
distance_prediction_head.eval()

# Load data
val_shard_dirs = [
    precomputed_small_emb_dir_val,
    precomputed_large_emb_dir_val,
]

val_logit_loader = PrecomputedShardLoader(
    val_shard_dirs, dataset_filter_path=val_dataset_filter_path
)

# We load the filter manually here to sync with the surrounding text
with open(val_dataset_filter_path, "rb") as fp:
    filter = pickle.load(fp)

# The surrounding data
with open(raw_data_json, "r") as fp:
    raw_data = json.load(fp)

count = 0
count_correct = 0
count_fake_correct = 0
count_zero = 0
se_zero = 0
se_one = 0
se_correct = 0
se_incorrect = 0
se = []
val_gt = []
for i, shard_tups in enumerate(val_logit_loader):
    if(i % 1000 == 0):
        print(f"Processed {i} examples...")

    small_tup, large_tup = shard_tups[:2]

    small_key, small_emb = small_tup
    large_key, large_emb = large_tup

    assert(small_key == large_key)
    assert(small_emb.shape[0] == large_emb.shape[0])

    if(small_emb.shape[0] == 0):
        continue

    # Compute the targets
    small_emb = small_emb.to(device=DEVICE, dtype=DTYPE)
    large_emb = large_emb.to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        # Compute logits from the small model embeddings
        small_logits = small_lm_head(small_emb)
        large_logits = large_lm_head(large_emb)

        small_logits_softmax = torch.nn.functional.softmax(small_logits, dim=-1)
        large_logits_softmax = torch.nn.functional.softmax(large_logits, dim=-1)

        small_logs = torch.nn.functional.log_softmax(small_logits, dim=-1)
        small_entropy = torch.sum(-1 * small_logits_softmax * small_logs, dim=-1)
        large_logs = torch.nn.functional.log_softmax(large_logits, dim=-1)
        large_entropy = torch.sum(-1 * large_logits_softmax * large_logs, dim=-1)

        targets = large_entropy

        # Discretize the target
        targets = discretize(
            targets,
            no_bins, 
            mi=min_bin, 
            ma=max_bin,
        )

    # Get the predictions
    predictions = distance_prediction_head(small_emb)
    predictions = torch.argmax(predictions, dim=-1)
    
    for s, g in zip(small_entropy, targets):
        se.append(s.item())
        val_gt.append(g.item())

    count_fake_correct += torch.sum((small_entropy >= 3.3) == targets).item()

    count += predictions.shape[0]
    count_correct += torch.sum(predictions == targets).item()
    count_zero += torch.sum(targets == 0).item()

    se_zero += torch.sum(small_entropy[targets == 0]).item()
    se_one += torch.sum(small_entropy[targets == 1]).item()

    se_correct += torch.sum(small_entropy[predictions == targets]).item()
    se_incorrect += torch.sum(small_entropy[predictions != targets]).item()

    # Load the surrounding data
    text = raw_data[small_key]
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    tokens = tokens[:MAX_LENGTH]

    # Load the filter
    filt = filter[small_key]

    assert(filt.shape == tokens.shape)
    assert(torch.sum(filt) == small_emb.shape[0])

    prediction_index = 0
    tokens_for_inspection = []
    for i, t in enumerate(tokens):
        tokens_for_inspection.append(tokenizer.decode(t))
        if(filt[i]):
            tokens_for_inspection.append(f"[{predictions[prediction_index]}, {targets[prediction_index]}]")
            prediction_index += 1

    print(f"{torch.sum(filt)} predictions...")
    print("".join(tokens_for_inspection))
    input("Press Enter to continue...")

print(f"Accuracy: {count_correct / count}")
print(f"Fake accuracy: {count_fake_correct / count}")
print(f"SE_zero: {se_zero / count_zero}")
print(f"SE_one: {se_one / (count - count_zero)}")
print(f"SE correct: {se_correct / count_correct}")
print(f"SE incorrect: {se_incorrect / (count - count_correct)}")

import pickle
with open("val_data.pickle", "wb") as fp:
    pickle.dump((se, val_gt), fp, protocol=pickle.HIGHEST_PROTOCOL)
