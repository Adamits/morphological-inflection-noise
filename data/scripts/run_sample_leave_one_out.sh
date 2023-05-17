#!/bin/bash

# Set K to a high number s.t. we sample all
K=100000000

annotations=("SLOT_ERROR" "MAPPED_POS_PAIR_ERROR" "POS_PAIR_ERROR" "POS_ERROR;SLOT_ERROR" "POS_ERROR;POS_PAIR_ERROR;SLOT_ERROR" "LEXICAL_ERROR" "PARADIGM_ERROR;SLOT_ERROR" "POS_PAIR_ERROR;SLOT_ERROR" "POS_ERROR;POS_PAIR_ERROR" "LEXICAL_ERROR;SLOT_ERROR" "POS_ERROR" "PARADIGM_ERROR")

for ann in ${annotations[@]}; do
    echo Making partition ${ann}
    for language in deu isl rus swe; do
        echo $language
        mkdir -p data/sampled/leave-one-out/${ann}
        
        python -u sample_final_dataset.py \
            --annotated_fn data/auto-annotated/${language}_auto_full.csv \
            --valid_tags_fn data/tUMPC/lexicon/${language}_valid_tags.txt \
            --output_fn data/sampled/leave-one-out/${ann}/${language}_sampled.tsv \
            --leave_out_annotations $ann \
            --k $K > data/sampled/leave-one-out/${ann}/sampled_${language}.log;
    done;
done