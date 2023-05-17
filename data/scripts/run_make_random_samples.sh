#! /bin/bash

# Makes 20 different random partitions of data, where we sample 75% of original dataset.

K=0.70
exp=baseline

for language in deu isl rus swe; do
    for i in {1..20}; do
        python -u noise-annotator/scripts/make_random_samples.py \
            --data-path noise-annotator/data/sampled/$exp/${language}_sampled.tsv \
            --k $K \
            --outfile noise-annotator/data/sampled/$exp/random_partitions/partition_${i}/${language}_sampled.tsv
    done ;
done