#!/bin/bash

# CURRENTLY USES JUST WORDS IN EXACTLY THE TRAINING PARTITION

readonly DATA_DIR=data/sampled/MSD-reinflection-resampled-leave-one-out
readonly OUT_DIR=data/sampled/pretraining/MSD-reinflection-resampled-leave-one-out

for language in deu isl rus swe; do
    for dir in $DATA_DIR/*; do
        annotation=$(basename $dir)
        python -u scripts/make_pretraining_data.py \
            --filename ${DATA_DIR}/${annotation}/${language}_sampled.tsv \
            --out-filename ${OUT_DIR}/${annotation}/${language}_sampled.tsv \
            --format pairs;
    done;
done
