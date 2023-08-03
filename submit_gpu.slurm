#!/bin/bash

#SBATCH -c 1 # Number of cores requested
#SBATCH -t 0-09:00 # Runtime in minutes
#SBATCH -p gpu # Partition to submit to
#SBATCH --mem=180000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab

module purge
module load Mambaforge
source activate hallucination

# create dataset filter for the repetition experiment
python create_dataset_filter.py /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/7B_test \
  /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_logits/30B_test \
  /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/test \
  --entropy_max 3.0

# reptition experiment
python repetition.py results/filter /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_test.json 7B 

# analyze results for the repetition experiment
 python generate_from_repetition.py  /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/test results