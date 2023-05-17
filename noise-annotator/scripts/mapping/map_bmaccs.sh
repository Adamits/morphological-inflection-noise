#!/bin/bash

for lang in deu fin isl rus swe; do
    echo "Mapping ${lang}..."
    python -u scripts/mapping/map_bmacc.py \
        --tumpc_fn data/tUMPC/lexicon/${lang}_tumpc.tsv \
        --gold_fn data/tUMPC/lexicon/${lang}_apertium_mapped.tsv \
        --outfn data/tUMPC/apt_uni-tumpc-mappings/${lang}.tsv ;
done