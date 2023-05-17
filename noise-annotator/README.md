# noise-annotator

## Scripts for using apertium FST morphological analyzers to automatically detect obvious noise in inflection pairs


## Pipeline
See pipeline.sh for the below steps. Running `pipeline.sh` should generate each step, and ultimately all of the successfully annotated data.

1. Produce a lexicon from tUMPC data with `scripts/make_tumpc_lexicon.py`. This produces a tsv of all words in the csv of tumpc inflection pairs. It has 2 columns: word, slot
2. Produce a lexicon of tUMPC data with the apertium analyses for each word with `scripts/make_apertium_lexicon.py`. This is in the same format at the tUMPC lexicon, except the 2nd column has comma-delimited analyses, since apertium generates all analyses.
3. Run `annotate.py` to get error annotations, except for slot errors.
4. Follow steps in `scripts/mapping` to produce a lexicon of tUMPC data where each word is assigned its UniMorph tags. Here instead of comma-delimited analyses, we have multiple entries for each word, each with exactly one tag. The mapping is found by mapping both UniMorph and tUMPC slots to apertium analyses, and then producing the lexicon with apertium as the ground truth, and assigning corresponding mapped UniMorph tags to tUMPC words.
5. Run `annotate_slots.py` to add slot error annotations to the annotated tUMPC pairs.

## Sampling training data
Run by `scripts/run_sample_final_dataset.sh`.

This gets the annotation distribution