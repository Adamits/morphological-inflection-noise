#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --output=logs/train_bilstm_attn_inflector.%j.log

source /curc/sw/anaconda3/latest
conda activate noisy-inflection

# Training for the bidirectional LSTM with attention.

set -eou pipefail

readonly ROOT="/rc_scratch/adwi9965/noisy-inflection"
readonly DATA="${ROOT}/data"
readonly ARCH=lstm
readonly RESULTS_PATH="${ROOT}/results/K-msd_resampled/${K}/${ARCH}-sigmorphon-resampled-reinflection"

# Model parameters from Kann & Schütze (MED):
# https://aclanthology.org/W16-2010/
readonly NUM_EPOCHS=60
readonly OPTIMIZER=adadelta
readonly LEARNING_RATE=1.0
readonly BATCH_SIZE=20
readonly EVAL_BATCH_SIZE=40
readonly EVAL_EVERY=2
readonly DROPOUT=0.3
readonly LAYERS=1
readonly EMBEDDING_SIZE=100
readonly HIDDEN_SIZE=100

readonly TRAIN="${DATA}/train/baseline/msd_sampled/${LANGUAGE}_sampled_sigmorphon_resampling_reinflection.tsv"
readonly DEV="${DATA}/test/${LANGUAGE}-dev"

# TODO: experiment with LR scheduler.
inflectors-train \
    --arch "${ARCH}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lang "${LANGUAGE}" \
    --experiment-name "${LANGUAGE}" \
    --target-col 2 \
    --features-col 4 \
    --train-data-path "${TRAIN}" \
    --dev-data-path "${DEV}" \
    --num-dataloader-workers 2 \
    --num-epochs "${NUM_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --embedding-size "${EMBEDDING_SIZE}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --enc-layers "${LAYERS}" \
    --dec-layers "${LAYERS}" \
    --optimizer "${OPTIMIZER}" \
    --max-decode-len 128 \
    --save-top-k 5 \
    --eval-every "${EVAL_EVERY}" \
    --output-path "${RESULTS_PATH}"