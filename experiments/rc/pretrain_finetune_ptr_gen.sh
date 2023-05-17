#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --output=logs/pretrain_fintune_ptr_gen_inflector.%j.log

source /curc/sw/anaconda3/latest
conda activate noisy-inflection

# Training for the ptr generator LSTM with attention.

set -eou pipefail

readonly ROOT="/rc_scratch/adwi9965/noisy-inflection"
readonly DATA="${ROOT}/data"
readonly ARCH=pointer_generator_lstm
readonly RESULTS_PATH="${ROOT}/results/${PARTITION}-pretrain-finetune/${K}/${ARCH}-attn"

# Model parameters from Sharma et. al. for high setting:
# https://aclanthology.org/K18-3013/
readonly NUM_EPOCHS=60
readonly NUM_PRETRAIN_EPOCHS=40
readonly OPTIMIZER=adam
readonly LEARNING_RATE=0.001
readonly BATCH_SIZE=32
readonly EVAL_BATCH_SIZE=64
readonly EVAL_EVERY=1
readonly DROPOUT=0.3
readonly LAYERS=1
readonly EMBEDDING_SIZE=300
readonly HIDDEN_SIZE=100

readonly TRAIN="${DATA}/train/${PARTITION}/${LANGUAGE}_sampled.tsv"
readonly DEV="${DATA}/test/${LANGUAGE}-dev"
readonly PRETRAIN="${DATA}/train/pretraining/${LANGUAGE}_sampled.tsv"

# TODO: experiment with LR scheduler.
# TODO: train/dev/test split.
yoyodyne-train \
    --arch "${ARCH}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lang "${LANGUAGE}" \
    --target-col 2 \
    --features-col 4 \
    --experiment-name "${LANGUAGE}" \
    --seed "${SEED}" \
    --train-data-path "${TRAIN}" \
    --dev-data-path "${DEV}" \
    --pretraining-data-file "${PRETRAIN}" \
    --dataloader-workers 2 \
    --epochs "${NUM_EPOCHS}" \
    --pretrain-epochs "${NUM_PRETRAIN_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --embedding-size "${EMBEDDING_SIZE}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --enc-layers "${LAYERS}" \
    --dec-layers "${LAYERS}" \
    --optimizer "${OPTIMIZER}" \
    --pretrain-scheduler warmupinvsqr \
    --pretrain-warmup-steps 100 \
    --gradient-clip 3.0 \
    --max-decode-len 128 \
    --save-top-k 5 \
    --eval-every "${EVAL_EVERY}" \
    --output-path "${RESULTS_PATH}"
