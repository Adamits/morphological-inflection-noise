#!/bin/bash

readonly CORRECTS_PATH=data/sampled/corrects
readonly NOISE_PARTS_PATH=data/sampled/baseline/random_noise_partitions
readonly OUT_DIR=data/sampled/baseline/random_augmented_noise

for language in deu isl rus swe; do
    python -u scripts/make_augmented_random_noisy_samples.py \
        --corrects-path $CORRECTS_PATH/${language}_sampled.tsv \
        --noise-partitions-path $NOISE_PARTS_PATH \
        --out-dir $OUT_DIR \
        --language $language \
        --num-partitions 10;
done
