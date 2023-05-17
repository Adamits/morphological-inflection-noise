#! /bin/bash

EXPERIMENT="corrects/msd_sampled-pretrain-finetune"

python -u scripts/make_dev_results_table.py \
    --results_dir "/rc_scratch/adwi9965/noisy-inflection/results" \
    --experiment $EXPERIMENT \
    --out_fn tables/results/${EXPERIMENT}/results.tsv