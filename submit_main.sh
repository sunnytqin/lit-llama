#!/bin/bash

# for n in {0..6}; 
# do
#     echo $n
#     sbatch --export=shard_count=$n submit_gpu.slurm 
#     sleep 2

# done


for n in {3..5}; 
do
    echo $n
    python repetition.py /n/holyscratch01/barak_lab/Lab/sqin/hallucination/repetition/wiki/1b_12b \
    $n \
    /n/holyscratch01/barak_lab/Lab/gahdritz/wikipedia/wiki_val.json \
    pythia 1b-deduped --sample_until_period=False \
    --addl_token_limit=100 \
    --experiment_name=repetition_default
done

