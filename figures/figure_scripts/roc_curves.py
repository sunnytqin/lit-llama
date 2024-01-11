import os
import sys
sys.path.append('.')

import json
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer

from train_head_utils import (
    discretize,
    DistancePredictionHead,
    load_pythia_model,
    load_lm_head,
    PrecomputedShardLoader,
    VOCAB_SIZES,
    _preprocessor,
)

torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda:0")
DTYPE = torch.float32
MAX_LENGTH = 2048

RUN_NAME = "random_seeds_2_pane"

KWARGS = []
labels = []
titles = []
pane_index = 0

# # 7B_30B
# KWARGS.append([])
# labels.append([])
# titles.append("LLaMA 7B/30B")

# run_name = "7B_30B_separated_classes"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/30B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_val/filter.pickle",
# })
# labels[pane_index].append("MLP")

# run_name = "7B_30B_separated_classes_linear"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 0,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/30B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_val/filter.pickle",
# })
# labels[pane_index].append("Linear classifier")

# run_name = "7B_30B_separated_classes_sme"
# KWARGS[pane_index].append({
#    "run_name": "7B_30B_separated_classes_sme",
#    "mode": "bet",
#    "head_checkpoint_path": f"outputs/7B_30B_separated_classes/state_dict.pth", # IGNORED
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/30B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_val/filter.pickle",
# })
# labels[pane_index].append("BET")

# run_name = "7B_30B_separated_classes_lm_head_finetune_1e_5"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "mode": "finetuned_lm_head_threshold",
#    "finetuned_lm_head": True,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/30B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_val/filter.pickle",
# })
# labels[pane_index].append("BET-FT")

# pane_index += 1

# # 7B_65B
# KWARGS.append([])
# labels.append([])
# titles.append("LLaMA 7B/65B")

# run_name = "7B_65B_separated_classes"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "65B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/65B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
# })
# labels[pane_index].append("MLP")

# run_name = "7B_65B_separated_classes_linear"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "65B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 0,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/65B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
# })
# labels[pane_index].append("Linear classifier")

# run_name = "7B_65B_separated_classes_sme"
# KWARGS[pane_index].append({
#    "run_name": "7B_65B_separated_classes_sme",
#    "mode": "bet",
#    "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth", # IGNORED
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "65B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/65B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
# })
# labels[pane_index].append("BET")

# run_name = "7B_65B_separated_classes_lm_head_finetune_1e_5"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "mode": "finetuned_lm_head_threshold",
#    "finetuned_lm_head": True,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "65B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/65B_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
# })
# labels[pane_index].append("BET-FT")

# pane_index += 1

# # pythia_1.4b_pythia_12b
# KWARGS.append([])
# labels.append([])
# titles.append("Pythia 1.4B/12B")

# run_name = "pythia-1.4b_pythia-12b_separated_classes"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "pythia",
#    "small_model_size": "1.4b",
#    "large_model_size": "12b",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-1.4b_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-12b_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_val/filter.pickle",
# })
# labels[pane_index].append("MLP")

# run_name = "pythia-1.4b_pythia-12b_separated_classes_linear"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "pythia",
#    "small_model_size": "1.4b",
#    "large_model_size": "12b",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 0,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-1.4b_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-12b_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_val/filter.pickle",
# })
# labels[pane_index].append("Linear classifier")

# run_name = "pythia-1.4b_pythia-12b_separated_classes_sme"
# KWARGS[pane_index].append({
#    "run_name": "7B_65B_separated_classes_sme",
#    "mode": "bet",
#    "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth", # IGNORED
#    "model_type": "pythia",
#    "small_model_size": "1.4b",
#    "large_model_size": "12b",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-1.4b_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-12b_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_val/filter.pickle",
# })
# labels[pane_index].append("BET")

# run_name = "pythia-1.4b_pythia-12b_separated_classes_lm_head_finetune"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "mode": "finetuned_lm_head_threshold",
#    "finetuned_lm_head": True,
#    "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
#    "model_type": "pythia",
#    "small_model_size": "1.4b",
#    "large_model_size": "12b",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-1.4b_val",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/pythia-12b_val",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_val/filter.pickle",
# })
# labels[pane_index].append("BET-FT")

# pane_index += 1

# # 7B_30B_OOD
# KWARGS.append([])
# labels.append([])
# titles.append("LLaMA 7B/30B (OOD)")

# run_name = "7B_30B_code_eval"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/7B_30B_separated_classes/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_code",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_code",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (MLP)")

# run_name = "7B_30B_europarl_eval"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/7B_30B_separated_classes/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_europarl",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_europarl",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (MLP)")

# run_name = "7B_30B_stack_exchange_eval"
# KWARGS[pane_index].append({
#    "run_name": run_name,
#    "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth",
#    "model_type": "llama",
#    "small_model_size": "7B",
#    "large_model_size": "30B",
#    "no_bins": 2,
#    "min_bin": 0,
#    "max_bin": 1,
#    "no_hidden_layers": 1,
#    "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#    "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#    "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_stack_exchange",
#    "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_stack_exchange",
#    "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (MLP)")

# run_name = "7B_30B_code_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_30B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "30B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_code",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_code",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (linear)")

# run_name = "7B_30B_europarl_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_30B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "30B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_europarl",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_europarl",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (linear)")

# run_name = "7B_30B_stack_exchange_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_30B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "30B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/30B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_stack_exchange",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/30B_pile_val_val_stack_exchange",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_30B_pile_val_val_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (linear)")

# pane_index += 1

# # 7B_65B_OOD
# KWARGS.append([])
# labels.append([])
# titles.append("LLaMA 7B/65B (OOD)")

# run_name = "7B_65B_code_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_code",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_code",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (MLP)")

# run_name = "7B_65B_europarl_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_europarl",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_europarl",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (MLP)")

# run_name = "7B_65B_stack_exchange_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_stack_exchange",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_stack_exchange",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (MLP)")

# run_name = "7B_65B_code_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_code",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_code",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (linear)")

# run_name = "7B_65B_europarl_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_europarl",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_europarl",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (linear)")

# run_name = "7B_65B_stack_exchange_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/7B_65B_separated_classes_linear/state_dict.pth",
#   "model_type": "llama",
#   "small_model_size": "7B",
#   "large_model_size": "65B",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B/lit-llama.pth",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/7B_pile_val_val_stack_exchange",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/65B_pile_val_val_stack_exchange",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_pile_val_val_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (linear)")

# pane_index += 1

# # pythia-1.4b_pythia-12b_OOD
# KWARGS.append([])
# labels.append([])
# titles.append("Pythia 1.4B/12B (OOD)")

# run_name = "pythia-1.4b_pythia-12b_code_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_code",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_code",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pile_pythia-1.4b_pythia-12b_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (MLP)")

# run_name = "pythia-1.4b_pythia-12b_europarl_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_europarl",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_europarl",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (MLP)")

# run_name = "pythia-1.4b_pythia-12b_stack_exchange_eval"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 1,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_stack_exchange",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_stack_exchange",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (MLP)")

# run_name = "pythia-1.4b_pythia-12b_code_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes_linear/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_code",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_code",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pile_pythia-1.4b_pythia-12b_val_val_code/filter.pickle",
# })
# labels[pane_index].append("Code (linear)")

# run_name = "pythia-1.4b_pythia-12b_europarl_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes_linear/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_europarl",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_europarl",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_europarl/filter.pickle",
# })
# labels[pane_index].append("Europarl (linear)")

# run_name = "pythia-1.4b_pythia-12b_stack_exchange_eval_linear"
# KWARGS[pane_index].append({
#   "run_name": run_name,
#   "head_checkpoint_path": f"outputs/pythia-1.4b_pythia-12b_separated_classes_linear/state_dict.pth",
#   "model_type": "pythia",
#   "small_model_size": "1.4b",
#   "large_model_size": "12b",
#   "no_bins": 2,
#   "min_bin": 0,
#   "max_bin": 1,
#   "no_hidden_layers": 0,
#   "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-1.4b",
#   "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/pythia_redownload/pythia-12b",
#   "precomputed_small_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-1.4b_val_val_stack_exchange",
#   "precomputed_large_emb_dir_val": f"/n/holyscratch01/barak_lab/Lab/gahdritz/pile/logits/pythia-12b_val_val_stack_exchange",
#   "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/pythia-1.4b_pythia-12b_stack_exchange/filter.pickle",
# })
# labels[pane_index].append("Stack Exchange (linear)")

# pane_index += 1

KWARGS.append([])
labels.append([])
titles.append("LLaMA 7B/65B MLP (different seeds)")

seeds = [
    19541,
    23143,
    9381,
    31454,
    4560,
    15777,
    654,
    30562,
    23948,
    4730,
]

for seed in seeds:
    run_name = f"7B_65B_separated_classes_{seed}"
    KWARGS[pane_index].append({
        "run_name": run_name,
        "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
        "model_type": "llama",
        "small_model_size": "7B",
        "large_model_size": "65B",
        "no_bins": 2,
        "min_bin": 0,
        "max_bin": 1,
        "no_hidden_layers": 1,
        "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
        "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B_dummy/lit-llama.pth",
        "precomputed_small_emb_dir_val": f"/n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/7B_val",
        "precomputed_large_emb_dir_val": f"/n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/65B_val",
        "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
    })
    labels[pane_index].append("")

pane_index += 1

# Random seeds
KWARGS.append([])
labels.append([])
titles.append("LLaMA 7B/65B Linear (different seeds)")

seeds = [
    23890,
    248,
    29972,
    14723,
    29980,
    10388,
    31163,
    7361,
    13662,
    13420,
]

for seed in seeds:
    run_name = f"7B_65B_separated_classes_linear_{seed}"
    KWARGS[pane_index].append({
        "run_name": run_name,
        "head_checkpoint_path": f"outputs/{run_name}/state_dict.pth",
        "model_type": "llama",
        "small_model_size": "7B",
        "large_model_size": "65B",
        "no_bins": 2,
        "min_bin": 0,
        "max_bin": 1,
        "no_hidden_layers": 0,
        "small_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/7B/lit-llama.pth",
        "large_checkpoint_path": f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B_dummy/lit-llama.pth",
        "precomputed_small_emb_dir_val": f"/n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/7B_val",
        "precomputed_large_emb_dir_val": f"/n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/65B_val",
        "val_dataset_filter_path": f"/n/holyscratch01/barak_lab/Lab/gahdritz/dataset_filters/7B_65B_val/filter.pickle",
    })
    labels[pane_index].append("")

pane_index += 1


def get_preds(
    *,
    head_checkpoint_path: str,
    model_type: str,
    small_model_size: str,
    large_model_size: str,
    no_bins: int,
    min_bin: int,
    max_bin: int,
    no_hidden_layers: int,
    small_checkpoint_path: str,
    large_checkpoint_path: str,
    precomputed_small_emb_dir_val: str,
    precomputed_large_emb_dir_val: str,
    val_dataset_filter_path: str,
    mode: str = None,
    **kwargs,
):
    print(head_checkpoint_path)

    small_lm_head = load_lm_head(small_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=small_model_size)
    large_lm_head = load_lm_head(large_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=large_model_size)

    torch.save(large_lm_head.state_dict(), "dummy.pth")

    # Load the model
    if(mode == "finetuned_lm_head_threshold"):
        checkpoint = torch.load(head_checkpoint_path, map_location=DEVICE)
        weight = checkpoint["weight"]
        model = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        model = model.to(DEVICE)
        with torch.no_grad():
            model.weight.copy_(weight)
    elif(mode == "bet"):
        # No need to load the model
        model = lambda x: x
    else:
        shared_head_params = {
            "no_bins": no_bins,
            "hidden_dim": 2048,
            "no_hidden_layers": no_hidden_layers,
            "dropout": 0.1,
            "activation": "relu",
        }
        model = DistancePredictionHead(
            input_dim=small_lm_head.weight.shape[1],
            **shared_head_params,
        )

        checkpoint = torch.load(head_checkpoint_path)

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # Try compiling first
            model = torch.compile(model)
            model.load_state_dict(checkpoint)
            

        model.to(DEVICE)

    # Load data
    val_shard_dirs = [
        precomputed_small_emb_dir_val,
        precomputed_large_emb_dir_val,
    ]

    val_logit_loader = PrecomputedShardLoader(
        val_shard_dirs, dataset_filter_path=val_dataset_filter_path
    )

    stash = {}
    data_gen = _preprocessor(
        shard_loader=val_logit_loader,
        small_lm_head=small_lm_head,
        large_lm_head=large_lm_head,
        model_type=model_type,
        no_bins=no_bins,
        min_bin=min_bin,
        max_bin=max_bin,
        min_entropy=None,
        max_entropy=None,
        provide_entropy_as_input=False,
        use_logits_as_input=False,
        softmax_input_logits=False,
        target_fn_name="large_entropy",
        bin_target=True,
        append_predicted_token_embedding=False,
        small_embedding_layer=False,
        device=DEVICE,
        dtype=DTYPE,
        _stash=stash,
    )

    probs = []
    targets = []
    for i, (small_emb, target) in enumerate(data_gen):
        prediction = model(small_emb)

        if(mode == "finetuned_lm_head_threshold"):
            prediction_softmax = torch.nn.functional.softmax(prediction, dim=-1)
            prediction_softmax_log = torch.nn.functional.log_softmax(prediction, dim=-1)
            prediction_entropy = -torch.sum(prediction_softmax * prediction_softmax_log, dim=-1)
            prediction = prediction_entropy
        else: # binary classifier
            prediction = torch.nn.functional.softmax(prediction, dim=-1)[..., 1]

        probs.extend([float(t.detach().data) for t in prediction.unbind()])
        targets.extend([int(t.detach().data) for t in target.unbind()])

    if(mode == "bet"):
        probs = stash["small_entropy"]

    return probs, targets


data = []
for i, kwargs in enumerate(KWARGS):
    l = []
    for j, k in enumerate(kwargs):
        probs, targets = get_preds(**k)
        l.append((probs, targets))

    data.append(l)

os.makedirs("pickles", exist_ok=True)
import pickle
with open(f"pickles/roc_data_{RUN_NAME}.pickle", "wb") as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"pickles/roc_data_{RUN_NAME}.pickle", "rb") as fp:
    data = pickle.load(fp)

all_plots = []
all_metrics = []
for plot_kwargs, plot_data in zip(KWARGS, data):
    plots = []
    metrics = []
    for kwargs, (probs, targets) in zip(plot_kwargs, plot_data):
        metric_dict = {}

        mode = kwargs.get("mode", None)

        sorted_pairs = list(sorted(zip(probs, targets), key=lambda x: x[0]))

        def compute_point(pos, neg):
            tp = 0
            fp = 0
            fn = 0
            tn = 0

            for p in pos:
                if(p == 1):
                    tp += 1
                else:
                    fp += 1

            for n in neg:
                if(n == 0):
                    tn += 1
                else:
                    fn += 1

            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)

            return fpr, tpr

        points = []
        best_accuracy = 0
        for i in range(len(sorted_pairs)):
            neg = [t[1] for t in sorted_pairs[:i]]
            pos = [t[1] for t in sorted_pairs[i:]]
            correct_neg = sum([1 for t in neg if t == 0])
            correct_pos = sum([1 for t in pos if t == 1])
            accuracy = (correct_neg + correct_pos) / len(sorted_pairs)
            if(accuracy > best_accuracy):
                best_accuracy = accuracy
            fpr, tpr = compute_point(pos, neg)
            points.append((fpr, tpr))

        auc = roc_auc_score(*reversed(list(zip(*sorted_pairs))))
        metric_dict["AUC"] = auc

        if(mode is None):
            accuracy = sum(
                [
                    1 for p, t in zip(probs, targets) 
                    if (p >= 0.5 and t == 1) or (p < 0.5 and t == 0)
                ]
            ) / len(probs)
        else:
            accuracy = best_accuracy

        metric_dict["Acc."] = accuracy

        plots.append(list(sorted(points, key=lambda x: x[0])))
        metrics.append(metric_dict)

    all_plots.append(plots)
    all_metrics.append(metrics)

no_plots = len(all_plots)
no_cols = min(3, no_plots)
no_rows = no_plots // no_cols + (no_plots % no_cols > 0)

fig = plt.figure(figsize=(6 * no_cols, 6 * no_rows))
axs = fig.subplots(no_rows, no_cols, sharex=True, sharey=True)
if(no_rows == 1):
    axs = [axs]

if(no_cols == 1):
    axs = [axs]

for i, (plots, metrics, label_set, title) in enumerate(zip(all_plots, all_metrics, labels, titles)):
    row_index = i // no_cols
    col_index = i % no_cols

    ax = axs[row_index][col_index]

    if(row_index == no_rows - 1):
        ax.set_xlabel("False Positive Rate")

    if(col_index == 0):
        ax.set_ylabel("True Positive Rate")

    ax.set_title(title)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.75)
    for label, points, metric in zip(label_set, plots, metrics):
        ax.plot(*zip(*points), label=f"{label} ({', '.join([f'{k}: {v:.2f}' for k, v in metric.items()])})")

    ax.legend()

plt.savefig(f"figures/roc_curve_{RUN_NAME}.pdf", bbox_inches="tight")
