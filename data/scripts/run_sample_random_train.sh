# Sample from the noise distribution to build a train dataset the same size as
# The number of corrects per language.

FILEPATH=noise-annotator/data/sampled
# For running on different sampling partitions
ext=sampled_sigmorphon_resampling_reinflection.tsv

for language in deu isl rus swe; do
    input_fp=${FILEPATH}/baseline/msd_sampled/${language}_${ext}
    corrects_fp=${FILEPATH}/corrects/msd_sampled/${language}_${ext}
    output_fp=${FILEPATH}/baseline_correct_size/${language}_${ext}

    K=$(wc -l ${corrects_fp} | cut -d ' ' -f5)
    echo ${input_fp} $K
    
    python -u noise-annotator/scripts/make_random_train_sample.py \
        --input-filepath ${input_fp} \
        --output-filepath ${output_fp} \
        --k $K ;
done