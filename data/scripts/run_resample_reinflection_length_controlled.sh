#!/bin/bash

UNI_PATH="/Users/adamwiemerslage/nlp-projects/morphology/unimorph"
DATAPATH="/Users/adamwiemerslage/nlp-projects/morphology/noisy-inflection/noise-annotator/data/sampled"

for lang in deu isl rus swe; do
    filename="${lang}_sampled_sigmorphon_resampling_reinflection"
    outfile="$DATAPATH/baseline/TEST_length_controlled/${filename}.tsv"
    logfile="$DATAPATH/baseline/TEST_length_controlled/${lang}.log"

    python noise-annotator/scripts/resample_reinflection_length_controlled.py \
        --tumpc-filepath noise-annotator/data/sampled/baseline/${lang}_sampled.tsv \
        --lang $lang \
        --apt-filepath noise-annotator/data/tUMPC/lexicon/${lang}_apertium.tsv \
        --unimorph-filepath "${UNI_PATH}/${lang}/${lang}"\
        --test-filepath inflection/data/test/${lang}-dev \
        --output-filepath $outfile  > $logfile;
done;