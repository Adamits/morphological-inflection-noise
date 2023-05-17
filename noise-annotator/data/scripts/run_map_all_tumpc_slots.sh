#! /bin/bash

for lang in deu isl rus swe; do
    python noise-annotator/data/scripts/map_tumpc_slots.py \
        --inflections-fn noise-annotator/data/tUMPC/full/${lang}.csv \
        --tumpc-map-fn noise-annotator/data/tUMPC/apt_uni-tumpc-mappings/${lang}.tsv \
        --out-fn noise-annotator/data/tUMPC/all_mapped/${lang}.tsv ;
done