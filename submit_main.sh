#!/bin/bash

for n in {1..8}; 
do
    echo $n
    sbatch --export=shard_count=$n submit_gpu.slurm 
    sleep 2

done