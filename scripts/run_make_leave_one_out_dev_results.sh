#! /bin/bash

for dir in data/train/MSD-reinflection-resampled-add-one-in/*; do 
# for dir in data/train/leave-one-out/*; do 
    EXPERIMENT=$(echo $dir | cut -d'/' -f3-4)
    python -u scripts/make_dev_results_table.py \
        --results_dir "/rc_scratch/adwi9965/noisy-inflection/results" \
        --experiment $EXPERIMENT \
        --out_fn tables/results/${EXPERIMENT}/results.tsv ;
done