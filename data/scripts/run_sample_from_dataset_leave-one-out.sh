#!/ bin/bash


annotations=("SLOT_ERROR" "MAPPED_POS_PAIR_ERROR" "POS_PAIR_ERROR" "POS_ERROR;SLOT_ERROR" "POS_ERROR;POS_PAIR_ERROR;SLOT_ERROR" "LEXICAL_ERROR" "PARADIGM_ERROR;SLOT_ERROR" "POS_PAIR_ERROR;SLOT_ERROR" "POS_ERROR;POS_PAIR_ERROR" "LEXICAL_ERROR;SLOT_ERROR" "POS_ERROR" "PARADIGM_ERROR")

for ann in ${annotations[@]}; do
    echo Making partition ${ann}
    for language in deu isl rus swe; do
        echo $language
        outdir=data/sampled/MSD-reinflection-resampled-leave-one-out/${ann}
        mkdir -p $outdir
        
        python -u scripts/sample_from_dataset.py \
            --data-path data/sampled/baseline/msd_sampled/${language}_sampled_sigmorphon_resampling_reinflection.tsv \
            --output-path ${outdir}/${language}_sampled.tsv \
            --leave_out_annotations $ann > ${outdir}/sampled_${language}.log;
    done;
done