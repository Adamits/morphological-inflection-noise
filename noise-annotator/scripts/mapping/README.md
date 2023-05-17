# Mapping
This is for the various parts of getting a shared slot/MSD space. This is a prerequisite for performing slot error annotation, and for evaluating the system.

Pipeline:
1. **Make sure you have tUMPC lexicon** (`scripts/mapping/make_tumpc_lexicons.sh`): This turns the paradigms into a single list of words and the slot they were marked with. If a word is marked with >1 slot, those each become a unique entry in the lexicon
2. **Make sure you have apertium lexicon** (`scripts/mapping/make_apertium_lexicons.sh`): This is the same list of tUMPC words, but along with the POS and tags in their morphological analysis.
    - **TODO**: We might also want lemma
3. **Find invalid apt tags** Run `scripts/mapping/make_valid_invalid_tags.sh` to identify apt analyses that cause a conflict with UniMorph
3. **Map apertium lexicon to UniMorph** (`scripts/mapping/apply_maps.sh`): We do this with manual mappings in data/tUMPC/um-apertium-mappings/${LANG}_map.tsv. This produces a new lexicon, where apt analyses are replaced by UniMorph tags.
    - Words with a set of apertium analyses that caused a conflict with UniMorph are given the ERR MSD
    - Words with no analysis or an analysis that mapped to no atomic UniMorph tags are given the UNK MSD
4. **Map tUMPC to UniMorph tags via apertium lexicon** (`scripts/mapping/map_bmaccs.sh`): Run map_bmacc.py. This optimizes for the mapping that results in the highest word overlap between between tUMPC slots and apertium UniMorph tags, where apertium can model all possible tags for a given word.
5. Now this can be passed to annotation pipeline for which we simply check if the tUMPC mapped slot is in the set of apertium (mapped to UniMorph) tags.