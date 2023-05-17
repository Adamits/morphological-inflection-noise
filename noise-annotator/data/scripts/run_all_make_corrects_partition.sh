#!bin/bash

readonly DIR=noise-annotator/data/sampled

# for data_file in $DIR/baseline/msd_sampled/*.tsv; do
#     FN=$(basename $data_file)
#     echo $FN
#     mkdir -p ${DIR}/corrects/msd_sampled
    
#     python -u noise-annotator/data/scripts/make_corrects_partition.py \
#         $data_file \
#         ${DIR}/corrects/msd_sampled/${FN}
# done

EXP=TEST_length_controlled
for data_file in $DIR/baseline/${EXP}/*.tsv; do
    FN=$(basename $data_file)
    echo $FN
    mkdir -p ${DIR}/corrects/${EXP}
    
    python -u noise-annotator/data/scripts/make_corrects_partition.py \
        $data_file \
        ${DIR}/corrects/${EXP}/${FN}
done