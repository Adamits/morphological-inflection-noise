#!/bin/bash

# Set K to a high number s.t. we sample all
K=100000000

for language in deu isl rus swe; do
    echo $language
    
    python -u sample_final_dataset.py \
        --annotated_fn data/auto-annotated/${language}_auto_full.csv \
        --valid_tags_fn data/tUMPC/lexicon/${language}_valid_tags.txt \
        --output_fn data/sampled/corrects/${language}_sampled.tsv \
        --include-annotations C \
        --k $K > data/sampled/corrects/sampled_${language}.log;
done