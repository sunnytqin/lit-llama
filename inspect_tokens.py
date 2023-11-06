import torch
from transformers import AutoTokenizer

from train_head_utils import (
    batch_loader,
    DistancePredictionHead,
    load_pythia_model,
    load_lm_head,
    PrecomputedShardLoader,
    _preprocessor,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float32

classifier_checkpoint = "outputs/pythia-1.4b_pythia-12b_separated_classes_init/state_dict.pth"
model_type = "pythia"
small_model_size = "1.4b"
large_model_size = "12b"
small_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/models/pythia/pythia-{small_model_size}"
large_checkpoint_path = f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-{large_model_size}"
precomputed_small_emb_dir_val = f"data/wikipedia_scratch/wiki_logits/pythia-{small_model_size}_val"
precomputed_large_emb_dir_val = f"data/wikipedia_scratch/wiki_logits/pythia-{large_model_size}_val"
precomputed_head_input_emb_dir_val = f"data/wikipedia_scratch/wiki_logits/pythia-{small_model_size}_init_val"
val_dataset_filter_path = f"dataset_filters/pythia-{small_model_size}_pythia-{large_model_size}_val/filter.pickle"

tokenizer= AutoTokenizer.from_pretrained(
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
distance_prediction_head = torch.compile(distance_prediction_head)

checkpoint = torch.load(classifier_checkpoint, map_location=DEVICE)
distance_prediction_head.load_state_dict(checkpoint)
distance_prediction_head.eval()

# Load data
val_shard_dirs = [
    precomputed_small_emb_dir_val,
    precomputed_large_emb_dir_val,
    precomputed_head_input_emb_dir_val,
]

val_logit_loader = PrecomputedShardLoader(
    val_shard_dirs, dataset_filter_path=val_dataset_filter_path
)

data_gen = _preprocessor(
    shard_loader=val_logit_loader,
    small_lm_head=small_lm_head,
    large_lm_head=large_lm_head,
    model_type=model_type,
    no_bins=2,
    min_bin=0,
    max_bin=2,
    min_entropy = None,
    max_entropy = None,
    provide_entropy_as_input = False,
    target_fn_name="large_entropy",
    bin_target=True,
    device=DEVICE,
)

bl = batch_loader(
    data_gen=data_gen,
    batch_size=1,
    skip_frac=0.,
)

input_linear = small_model.gpt_neox.embed_in.weight

corrects = {}
zeros = {}
ones = {}
c = 0
total = 0
for i, (inp, target) in enumerate(bl):
    diffs = torch.sum(torch.abs(input_linear - inp), dim=1)
    decoded_id = torch.argmin(diffs).item()
    decoded = tokenizer.decode([decoded_id])
    
    prediction = torch.argmax(distance_prediction_head(inp)[0]).item()
    if(prediction == target.item()):
        corrects.setdefault(decoded, 0)
        corrects[decoded] += 1
        c += 1

    if(target.item() == 0):
        zeros.setdefault(decoded, 0)
        zeros[decoded] += 1
    elif(target.item() == 1):
        ones.setdefault(decoded, 0)
        ones[decoded] += 1

    total += 1

print(list(sorted(corrects.items(), key=lambda x: x[1], reverse=True)))
print(list(sorted(zeros.items(), key=lambda x: x[1], reverse=True)))
print(list(sorted(ones.items(), key=lambda x: x[1], reverse=True)))
print(f"Accuracy: {c / total}")
