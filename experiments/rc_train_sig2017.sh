#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "ERROR: expecting an argument: PARTITION"
    exit 1
fi

PARTITION=$1
ROOT=/rc_scratch/adwi9965
EXP_DIR=${ROOT}/noisy-inflection/experiments

# baseline partition
for language in deu isl rus swe; do
    for model in M_C; do #bilstm_attn ptr_gen transformer M_C; do
        echo $language, $PARTITION
        sbatch --export=ALL,PARTITION=$PARTITION,LANGUAGE=${language} ${EXP_DIR}/rc/sigmorphon2017/train_${model}.sh;
    done
done
