#! /bin/bash

DATADIR="data/tUMPC/lexicon"
OUTDIR="data/tUMPC/lexicon"

for l in deu fin isl rus swe; do
    analyzer="../apertium-analyzers/apertium-${l}/${l}.automorf.bin"

    if [ ! -f $analyzer ]; then
        analyzer="../apertium-analyzers/apertium-${l}/${l}.automorf.hfst"
    fi

    echo "Running ${l}"
    echo $analyzer
    python -u scripts/make_apertium_lexicon.py \
        --lexicon_fn "${DATADIR}/${l}_tumpc.tsv" \
        --analyzer_fn $analyzer \
        --outfn "${OUTDIR}/${l}_apertium.tsv" ;
done