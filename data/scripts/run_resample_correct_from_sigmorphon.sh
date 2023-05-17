#!/bin/bash

exp=baseline/msd_sampled
SIGMORPHON_DATA_PATH="/Users/adamwiemerslage/nlp-projects/morphology/sigmorphon/sigmorphon-data/data"
INFLECTION_DATA_PATH="/Users/adamwiemerslage/nlp-projects/morphology/noisy-inflection/inflection/data/"

for i in deu,german isl,icelandic rus,russian, swe,swedish; do IFS=","
    set -- $i
    language=$1
    sig_language=$2

    python -u scripts/resample_correct_from_sigmorphon.py \
        --samples_path data/sampled/$exp/${language}_sampled.tsv \
        --sigmorphon_path ${SIGMORPHON_DATA_PATH}/${sig_language}-train-high \
        --test_path ${INFLECTION_DATA_PATH}/test/${sig_language}-dev \
        --apertium_lexicon_path data/tUMPC/lexicon/${language}_apertium.tsv \
        --output_path data/sampled/$exp/${language}_sampled_sigmorphon_resampling.tsv > data/sampled/$exp/resample_sigmorphon_${language}.log
done