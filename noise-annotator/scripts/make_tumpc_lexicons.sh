#! /bin/bash

DATADIR="data/tUMPC/full"
OUTDIR="data/tUMPC/lexicon"

for l in deu fin isl rus swe; do
    python -u scripts/make_tumpc_lexicon.py \
        --fn "${DATADIR}/${l}.csv" \
        --outfn "${OUTDIR}/${l}_tumpc.tsv" ;
done