#!/bin/bash

ROOT="/Users/adamwiemerslage/nlp-projects/morphology"

for lang in deu fin isl rus swe; do
    echo $lang
    ext="bin"
    if [[ $lang = "fin" ]]
    then
        ext="hfst"
    fi

    python -u scripts/mapping/make_valid_invalid_tags.py \
        --map_fn data/tUMPC/um-apertium-mappings/${lang}_map.tsv \
        --uni_fn ${ROOT}/unimorph/${lang}/${lang} \
        --apertium_fn ${ROOT}/noisy-inflection/apertium-analyzers/apertium-${lang}/${lang}.automorf.${ext} \
        --outpath data/tUMPC/lexicon ;
done;