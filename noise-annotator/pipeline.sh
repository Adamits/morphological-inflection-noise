#!/bin/bash

# 1. Make a lexicon from tUMPC words
bash scripts/make_tumpc_lexicons.sh

# 2. Make a lexicon from tUMPC words with apertium analyses
bash scripts/make_apertium_lexicons.sh

# 3. Save the apertium analyses that lead to disagreement with UniMorph. These will be considered errors
bash scripts/mapping/make_valid_invalid_tags.sh

# 4. Make a lexicon of tUMPC words with UniMorph tags that were mapped to apertium, using the manual maps in data/tUMPC/um-apertium-mappings
bash scripts/mapping/apply_maps.sh

# 5. Map the tUMPC slots to UniMorph tags based on bmACC allignments to create 
#    the mapping files in data/tUMPC/apt_uni-tumpc-mappings
bash scripts/mapping/map_bmaccs.sh

# 6. Annotate all. 
#   (This used to not do slot errors, has since been updated to annotate everu category)
bash scripts/annotate_all.sh

# 7. Sample data
bash run_sample_final_dataset.sh

# 8. Resample replacing corrects with data from SIGMORPHON train split
bash run_resample_correct_from_sigmorphon.sh