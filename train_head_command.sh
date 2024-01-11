#!/bin/bash

source hallucinations_uncertainty_venv/bin/activate

RUN_NAME="7B_65B_separated_classes"

python3 train_head.py \
    /n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/7B \
    /n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/65B \
    "outputs/${RUN_NAME}" \
    --large_model_size 65B \
    --large_checkpoint_path /n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/65B_dummy/lit-llama.pth \
    --precomputed_small_emb_dir_val /n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/7B_val \
    --precomputed_large_emb_dir_val /n/holylabs/LABS/barak_lab/Lab/gahdritz/pickles/65B_val \
    --target_fn_name large_entropy \
    --min_bin 0 \
    --max_bin 1 \
    --no_bins 2 \
    --skip_frac 0. \
    --lr 1e-4 \
    --no_hidden_layers 1 \
    --no_epochs 1000 \
    --eval_every_n_batches 200 \
    --dataset_filter dataset_filters/7B_65B/filter.pickle \
    --val_dataset_filter dataset_filters/7B_65B_val/filter.pickle \
    --use_wandb True \
    --wandb_project hallucinations_uncertainty \
    --wandb_entity gwa2107 \
    --wandb_run_name "${RUN_NAME}"
