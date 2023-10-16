#!/bin/bash

#SBATCH -c 1 # Number of cores requested
#SBATCH -t 0-02:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=80000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab

module purge
module load Mambaforge
source activate hallucination

# precompute logits
python precompute_logits.py /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_val.json  \
  /n/holyscratch01/barak_lab/Lab/sqin/hallucination/wiki/logits/ \
  --model_type pythia  --model_size '12b-deduped' \
  --output_shard_size 1000 --return_embeddings True

# create dataset filter for the repetition experiment
python create_dataset_filter.py /n/holyscratch01/barak_lab/Lab/sqin/hallucination/wiki/logits/pythia-1b-deduped_val_val \
  /n/holyscratch01/barak_lab/Lab/sqin/hallucination/wiki/logits/pythia-12b-deduped_val_val \
  /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/wiki/1b_12b \
  /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_val.json \
  --model_type pythia --entropy_max 3.0 --small_model_size '1b-deduped' --large_model_size '12b-deduped' \
  --shard_output=True --balanced_classes=True

# reptition experiment
# echo "python repetition.py /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/val \
#    ${shard_count} \
#    /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_val.json \
#    llama 7B --sample_until_period=False \
#    --addl_token_limit=100 \
#    --experiment_name=repetition_no_sample_until_period_no_eos_token"

python repetition.py /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/wiki/7B_30B \
   0 \
   /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_val.json \
   llama 7B --sample_until_period=False \
   --addl_token_limit=100 \
   --experiment_name=repetition_default


# analyze results for the repetition experiment
 python generate_from_repetition.py  \
    /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/wiki/7B_30B/repetition_default \
    llama 7B