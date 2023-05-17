#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --output=logs/train_sigmorphon_2017_M_C_inflector.%j.log

source /curc/sw/anaconda3/latest
conda activate py2-7

# Training for the makarov and clematide IL model.
ORIGINAL_DIR=$(pwd)
LIB=/projects/adwi9965/paradigm-completion/lib/emnlp2018-imitation-learning-for-neural-morphology/lib

set -eou pipefail

readonly ROOT="/rc_scratch/adwi9965/noisy-inflection"
readonly DATA="${ROOT}/data"
readonly ARCH=M_C

readonly TRAIN="${DATA}/train/${PARTITION}/${LANGUAGE}-train-low"
readonly DEV="${DATA}/test/${LANGUAGE}-dev"
readonly RESULT="${ROOT}/results/${PARTITION}/M_C-sigmorphon-2017/${LANGUAGE}-low"

cd $LIB
# Hyperparams from 2018 IL paper
python -u run_transducer.py --dynet-seed 1 --dynet-mem 700 \
    --transducer=haem --sigm2017format \
    --input=100 --feat-input=20 --action-input=100 --pos-emb \
    --enc-hidden=200 --dec-hidden=200 --enc-layers=1 \
    --dec-layers=1   --mlp=0 --nonlin=ReLU --il-optimal-oracle \
    --il-loss=softmax-margin --il-beta=1 --beta=5  --il-k=12 --il-global-rollout \
    --dropout=0 --optimization=ADADELTA --l2=0  --batch-size=1 \
    --decbatch-size=1  --patience=20 --epochs=60 \
    --tag-wraps=both --param-tying  --mode=il   --beam-width=0 --beam-widths=4 \
    $TRAIN  $DEV  $RESULT ;

echo done...
cd $ORIGINAL_DIR