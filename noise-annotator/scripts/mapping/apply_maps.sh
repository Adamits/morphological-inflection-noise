#!/bin/bash

DATA=data/tUMPC

for lang in deu fin isl rus swe; do
    python -u scripts/mapping/apply_map.py \
        --map_fn $DATA/um-apertium-mappings/${lang}_map.tsv \
        --apertium_fn $DATA/lexicon/${lang}_apertium.tsv \
        --invalid_tags_fn $DATA/lexicon/${lang}_invalid_tags.txt \
        --valid_tags_fn $DATA/lexicon/${lang}_valid_tags.txt \
        --out_fn $DATA/lexicon/${lang}_apertium_mapped.tsv;
done