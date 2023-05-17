#!/bin/bash

readonly CORRECTS_PATH=data/sampled/corrects/msd_sampled
readonly NOISE_PARTS_PATH=data/sampled/baseline/random_tumpc_corrects_partitions
readonly OUT_DIR=data/sampled/baseline/msd_sampled/random_resampled_tumpc_corrects

for language in deu isl rus swe; do
    python -u scripts/make_random_resmpled.py \
        --corrects-path $CORRECTS_PATH/${language}_sampled_sigmorphon_resampling_reinflection.tsv \
        --noise-partitions-path $NOISE_PARTS_PATH \
        --out-dir $OUT_DIR \
        --language $language \
        --num-partitions 10;
done
