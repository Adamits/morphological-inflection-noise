#!/bin/bash

ROOT=/rc_scratch/adwi9965
EXP_DIR=${ROOT}/noisy-inflection/experiments

for language in isl swe rus; do
    for arch in ptr_gen bilstm_attn transformer M_C; do
        for this_k in 1 2 3 4 5 6 7 8 9 10; do
            echo $this_k
            sbatch --export=ALL,K=$this_k,LANGUAGE=$language ${EXP_DIR}/rc/train_${arch}_kth_result.sh;
        done;
    done;
done
