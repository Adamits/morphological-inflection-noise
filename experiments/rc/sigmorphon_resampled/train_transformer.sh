#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --constraint=Tesla
#SBATCH --gres=gpu:1
#SBATCH --qos=preemptable
#SBATCH --output=logs/train_sigmorphon_resampled_transformer_inflector.%j.log


source /curc/sw/anaconda3/latest
conda activate noisy-inflection

# Training for the transformer.

set -eou pipefail

readonly ROOT="/rc_scratch/adwi9965/noisy-inflection"
readonly DATA="${ROOT}/data"
readonly ARCH=transformer
readonly RESULTS_PATH="${ROOT}/results/${PARTITION}/${ARCH}-sigmorphon-resampled"

# Model parameters from Wu et al. ("A Smaller Transformer"):
# https://aclanthology.org/2021.eacl-main.163/
readonly NUM_EPOCHS=800 #TODO: Lower? Dont want to overfit...
readonly EVAL_EVERY=16
readonly OPTIMIZER=adam
readonly LEARNING_RATE=0.001
readonly BATCH_SIZE=400
readonly EVAL_BATCH_SIZE=400
readonly DROPOUT=0.3
readonly ENC_LAYERS=4
readonly DEC_LAYERS=4
readonly EMBEDDING_SIZE=256
readonly HIDDEN_SIZE=1024
readonly NHEAD=4
readonly SMOOTHING=0.1
readonly SCHEDULER=warmupinvsqr
readonly WARMUP_STEPS=4000
readonly BETA2=0.98

readonly TRAIN="${DATA}/train/${PARTITION}/${LANGUAGE}_sampled_sigmorphon_resampling.tsv"
readonly DEV="${DATA}/test/${LANGUAGE}-dev"

# TODO: train/dev/test split.
python -u -m inflectors.train \
  --arch "${ARCH}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lang "${LANGUAGE}" \
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
  --enc-layers "${ENC_LAYERS}" \
  --dec-layers "${DEC_LAYERS}" \
  --nhead "${NHEAD}" \
  --optimizer "${OPTIMIZER}" \
  --gradient-clip 1.0 \
  --smoothing "${SMOOTHING}" \
  --beta2 "${BETA2}" \
  --lr-scheduler "${SCHEDULER}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --experiment-name "${LANGUAGE}" \
  --max-decode-len 128 \
  --save-top-k 5 \
  --eval-every "${EVAL_EVERY}" \
  --output-path "${RESULTS_PATH}"
