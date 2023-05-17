# Morphological Inflection Noise

Code, data, experiments, and results for **An Investigation of Noise in Morphological Inflection**.

## Repository Contents
### data
Training splits for all experiments. We also have the dev and test data from SIGMORPHON 2017, though only dev was used for this work.

### noise-annotator
The pipeline for annotating training samples from tUMPC. See the `noise-annotator/README.md`.

### noisy-yoyodyne
An old fork of yoyodyne updated for our experiments with noise and pretraining.

### scripts
Various scripts for evaluating and aggregating results

### tables
TSVs logging the stats on the best accuracy epoch for every training partition.