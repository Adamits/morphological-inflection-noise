#!/bin/bash

# REQUIRES CODE FORM https://github.com/Adamits/tUMPC
TUMPC_SCRIPT=tumpc/src/pos_based.py
CLUSTERS_DIR=tumpc/data/clusters/noisy-infl-mccurdy
CORPUS_DIR=tumpc/data/bible

for l in Icelandic German Russian Swedish Finnish; do
    python -u $TUMPC_SCRIPT \
        --clusters ${CLUSTERS_DIR}/${l}.clustered \
        --corpus ${CORPUS_DIR}/${l}.uppercase.txt \
        --output data/tumpc/full/${l};
done

rm -f data/tumpc/full/${l}.infl*