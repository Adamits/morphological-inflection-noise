# Noise Annotator
Scripts for generating the data pipeline. Assumes you have run tUMPC and converted the outputs to CSV. The results of that step are in `data/tUMPC/full/${language}.csv`

## Pipeline
See pipeline.sh for the steps to run. Running `pipeline.sh` should generate each step, and ultimately all of the successfully annotated data. This pipeline results in several intermediary representations:

- `data/tUMPC/lexicon/${language}_tumpc.tsv`: The lexicon of each word assigned a slot by tUMPC, and its corresponding slot.
- `data/tUMPC/lexicon/${language}_invalid_tags.txt`: The apertium analyses that could not be mapped, and are thus deemed invalid.
- `data/tUMPC/lexicon/${language}_valid_tags.txt`: The valid UniMorph tags/
- `data/tUMPC/lexicon/${language}_apertium_mapped.tsv`: The tUMPC lexicon with each word pointing to its gold UniMorph MSD according to the mapping from apertium to UniMorph. Each line is a unique word and MSD pairing, s.t. words mapped to more than one MSD entail more than one line.
- `data/tUMPC/apt_uni-tumpc-mappings/${language}.tsv`: A mapping from tUMPC slot to UniMorph MSD. THis is a 1:1 mapping in which the MSD that maximizes the F1 between tUMPC slots and MSDs according to those words that are in the silver-standard tags in the mapped apertium.
- `data/auto-annotated/${language}_auto_full.csv`: The full annotated data from tUMPC, including slot, msd, and a noise annotation according tot he pipeline.
- `data/sampled/baseline/${language}_sampled.tsv`: A training data file with all data that has a valid annotation. Note our codebase in `noisy-yoyodyne` is setup to ignore the 3rd and 5th columns, which contain the src MSD, and the noise annotation, respectively.

## Sampling training data
`scripts/run_sample_final_dataset.sh`, gets the annotation distribution, and contains examples for sampling other trianing partitions. See `data/scripts` for several related scripts.