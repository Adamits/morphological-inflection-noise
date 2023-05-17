#!/bin/bash
# This script is for training on subsplits in a partition

if [ "$#" -ne 1 ]; then
    echo "ERROR: expecting an arguments: DIR$"
    exit 1
fi

PARTITION=$1
ROOT=/rc_scratch/adwi9965
EXP_DIR=${ROOT}/noisy-inflection/experiments

for language in deu isl rus swe; do
    for model in bilstm_attn ptr_gen transformer M_C; do
        for dir in data/train/$PARTITION/*; do
            # Get PARTITION/annotation dir
            SUBPARTITION=$(echo $dir | cut -d'/' -f3-4)
            echo $language, $SUBPARTITION
            sbatch --export=ALL,PARTITION=$SUBPARTITION,LANGUAGE=${language} ${EXP_DIR}/rc/train_${model}.sh ;
        done ;
    done
done
