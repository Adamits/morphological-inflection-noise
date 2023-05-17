#!/bin/bash

exp=baseline/msd_sampled
UNI_PATH="/Users/adamwiemerslage/nlp-projects/morphology/unimorph"

for language in deu isl rus swe; do
    python -u scripts/make_sigmorphon_reinflection.py \
        --filename data/sampled/$exp/${language}_sampled_sigmorphon_resampling.tsv \
        --unimorph_filename ${UNI_PATH}/${language}/${language} \
        --output_filename data/sampled/$exp/${language}_sampled_sigmorphon_resampling_reinflection.tsv
done