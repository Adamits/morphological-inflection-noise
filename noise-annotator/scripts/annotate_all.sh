#!/bin/bash

for lang in fin; do #deu isl rus swe; do #fin; do
    echo $lang
    ext="bin"
    if [[ $lang = "fin" ]]
    then
        ext="hfst"
    fi

    python -u annotate.py \
        --anns_fn data/tUMPC/full/${lang}.csv \
        --apertium_fn data/tUMPC/lexicon/${lang}_apertium.tsv \
        --apertium_mapped_fn data/tUMPC/lexicon/${lang}_apertium_mapped.tsv \
        --tumpc_map_fn data/tUMPC/apt_uni-tumpc-mappings/${lang}.tsv \
        --wiki_lexerrors_fn data/wiki_filtered/${lang}.lex_errors \
        --outfn data/auto-annotated/${lang}_auto_full.csv \
        --lang $lang;
done

# For the 2k sample
# for lang in deu fin isl rus swe; do
#     echo $lang
#     ext="bin"
#     if [[ $lang = "fin" ]]
#     then
#         ext="hfst"
#     fi

#     python annotate.py \
#         --anns_fn data/tUMPC/full/${lang}_2k.csv \
#         --apertium_fn data/tUMPC/lexicon/${lang}_apertium.tsv \
#         --outfn data/auto-annotated/${lang}_auto_2k.csv \
#         --lang $lang;
# done