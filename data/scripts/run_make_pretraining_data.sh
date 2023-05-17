#!/bin/bash

# CURRENTLY USES JUST WORDS IN THE TUMPC BASELINE ANNOTATED SAMPLE

readonly DATA_DIR=data/sampled/extra_noise_partitions/baseline/random_resampled_noise
readonly OUT_DIR=data/sampled/pretraining/extra_noise_partitions/baseline/random_resampled_noise

for language in deu isl rus swe; do
    for dir in ${DATA_DIR}/${language}/*; do
        pct=$(basename $dir)
        out=${OUT_DIR}/${language}/$pct
        echo $dir $pct $out
        python -u scripts/make_pretraining_data.py \
            --filename ${dir}/${language}_sampled.tsv \
            --out-filename ${out}/${language}_sampled.tsv \
            --format pairs
    done
done
