#!bin/bash

readonly ORIGINAL_DIR=noise-annotator/data/sampled/ORIGINAL_ERRORS
readonly NEW_DIR=noise-annotator/data/sampled

for dir in $ORIGINAL_DIR/*; do
    PARTITION=$(basename $dir)
    for data_file in $dir/*.tsv; do
        echo $data_file
        echo $FN
        FN=$(basename $data_file)
        mkdir -p ${NEW_DIR}/${PARTITION}
        
        python -u noise-annotator/data/scripts/make_spurious_noise_corrects.py \
            $data_file \
            ${NEW_DIR}/${PARTITION}/${FN}
    done
done