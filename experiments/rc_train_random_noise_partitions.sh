#!/bin/bash

ROOT=/rc_scratch/adwi9965
EXP_DIR=${ROOT}/noisy-inflection/experiments
BASE_PARTITION=baseline/random_augmented_noise

# baseline partition
for language in deu isl rus swe; do
    for model in bilstm_attn ptr_gen transformer M_C; do
        for partition in {1..10}; do
            echo $language $model $partition
            P="${BASE_PARTITION}/${partition}"
            for k in 1 2 3 4 5; do
                SEED=$((42 * k))
                sbatch --export=ALL,PARTITION=$P,LANGUAGE=${language},SEED=${SEED},K=$k ${EXP_DIR}/rc/train_${model}.sh;
            done
        done
    done
done
