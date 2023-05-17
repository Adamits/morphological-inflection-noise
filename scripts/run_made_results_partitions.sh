#! /bin/bash

EXP=baseline/msd_sampled/random_resampled_noise

for dir in results/$EXP/*; do  
    echo $dir
    python -u scripts/make_dev_results_table.py \
        --results_dir "/rc_scratch/adwi9965/noisy-inflection" \
        --experiment $dir \
        --out_fn tables/${dir}/results.tsv ;
done