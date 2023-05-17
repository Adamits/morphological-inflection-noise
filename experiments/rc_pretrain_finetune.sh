#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "ERROR: expecting an argument: PARTITION"
    exit 1
fi

PARTITION=$1
ROOT=/rc_scratch/adwi9965
EXP_DIR=${ROOT}/noisy-inflection/experiments

# baseline partition
for language in deu; do #deu isl rus swe; do
    for model in ptr_gen; do #bilstm_attn ptr_gen transformer M_C; do
        echo $language $model $partition
        for k in 1; do #1 2 3 4 5; do
            SEED=$((42 * k))
            sbatch --export=ALL,PARTITION=$PARTITION,LANGUAGE=${language},SEED=${SEED},K=$k ${EXP_DIR}/rc/pretrain_finetune_${model}.sh;
        done
    done
done
