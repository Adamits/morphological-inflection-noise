import click
from typing import Dict, List, Set, Tuple
import random
# Set random seed for reproducibility.
random.seed(2022)

from collections import Counter

"""Here we replace the samples in samples_path annotated as correct with
pairs from SIGMORPHON training data.

Our goal is to 

1) sample as many of the same lemma, and same tags as possible
2) keep the lemma overlap % with the test data the same.
"""

# TODO: Consider computing the number of seen lemmas that are the SAME.
#       then we can resample to get the exact same test set coverage.

POS_MAP = {
    "vblex": "V",
    "vaux": "V",
    "vbmod": "V",
    "vbser": "V",
    "vbhaver": "V",
    "n": "N",
    "np": "PROPN",
    "adj": "ADJ",
    "adv": "ADV",
    "prn": "PRN",
    "num": "NUM",
    "det": "DET",
    "pr": "PREP"
}

def map_apert_pos(apert_pos: str) -> str:
    out = POS_MAP.get(apert_pos)

    if out == None:
        return apert_pos

    return out


def get_sig_POS(msd: str) -> str:
    return msd.split(";")[0]


# def get_from_apertium_lexicon(s: List, lex: Dict):
#     try:
#         return tuple(lex[(s[0].lower(), get_sig_POS(s[2]))])
#     except KeyError:
#         print("No such lexeme in apertium:")
#         print((s[0].lower(), get_sig_POS(s[2])))


# def get_samples_lemma_counts(samples: List, lex: Dict) -> Dict:
#     """Get the counts of lemmas in the List of samples from our data

#     Args:
#         samples (List): the samples comprising the words, msds and annotation
#         lex (Dict): Apertium lexiconmapping word to its possible lemma, pos tuples

#     Returns:
#         Dict: dicitonary of lemma counts
#     """
#     counts = {}
#     for s in samples:
#         # Get the lemmas for the src word
#         # TODO: For now we assume exactly one lemma per form
#         # TODO: We lowercase for now, update once we have correctly cased apertium lexicon
#         lemma_poses = lex[s[0]]
#         if lemma_poses is None:
#             continue

#         counts.setdefault(lemmas, 0)
#         counts[lemmas] += 1

#     return counts


def get_samples_MSD_counts(samples: List) -> Dict:
    """Count the (target) MSDs in the sample data

    Args:
        samples (List): he samples comprising the words, msds and annotation

    Returns:
        Dict: count of each MSD
    """
    counts = {}
    for src, tgt, src_msd, tgt_msd, annotation in samples:
        # Only tgt MSD is in SIGMORPHON
        counts.setdefault(tgt_msd, 0)
        counts[tgt_msd] += 1

    return counts


def read_apertium_lexicon(fn: str) -> Dict:
    lexicon = {}
    with open(fn, "r") as f:
        for line in f:
            word, analyses_str = line.rstrip().split("\t")
            analyses = analyses_str.split(",")
            # lemmas = list(set([analysis.split(";")[0] for analysis in analyses]))
            duplicates = set()
            for analysis in analyses:
                fields = analysis.split(";")
                lemma = fields[0]
                try:
                    pos = map_apert_pos(fields[1])
                except IndexError:
                    pos = "*"
                if (lemma, pos) in duplicates:
                    continue

                lexicon.setdefault(word, []).append((lemma, pos))
                duplicates.add((lemma, pos))

    return lexicon


def read_samples(fn: str) -> List:
    samples = []
    with open(fn, "r") as f:
        for line in f:
            src, tgt, src_msd, tgt_msd, annotation = line.rstrip().split("\t")
            samples.append((src, tgt, src_msd, tgt_msd, annotation))

    return samples


def read_sigmorphon(fn: str, original_msds: Set) -> Dict:
    """Read a dictionary of {lemma/pos: [triples in the paradigm]}.

    Ignore any triples with an MSD not in the original MSDs

    Args:
        fn (str): _description_
        original_msds (Set): _description_

    Returns:
        Dict: _description_
    """
    sigmorphon = {}
    with open(fn, "r") as f:
        for line in f:
            lemma, tgt, msd = line.rstrip().split("\t")
            if msd not in original_msds:
                continue
            
            pos = get_sig_POS(msd)
            # DO NOT SAMPLE PAIRS WITH A WHITESPACE-DELIMITED WORD IN IT
            if " " not in lemma and " " not in tgt:
                sigmorphon.setdefault((lemma, pos), []).append((lemma, tgt, msd))

    return sigmorphon


def sample_sigmorphon_matches(
    sigmorphon_dict: Dict,
    corrects: List,
    lex: Dict,
    sample_msd_counts: Dict
):
    samples = []
    sigmorphon_lemma_counts = {}
    sigmorphon_msd_counts = {}
    # Will be built with the leftover samples after sampling lemmas
    # This is to avoid resampling duplicates
    sigmorphon_msd_dict = {}
    # TODO: This should have EVERYTHING not sampled from SIGMORPHON
    leftovers = [s for _, samples in sigmorphon_dict.items() for s in samples]

    def _update_lemma_counts(new_samples):
        for lemma, tgt, msd in new_samples:
            sigmorphon_lemma_counts.setdefault(lemma, 0)
            sigmorphon_lemma_counts[lemma] += 1

    def _update_sig_msd_dict(samples):
        for src, tgt, msd in samples:
            sigmorphon_msd_dict.setdefault(msd, []).append((src, tgt, msd))

    # for lemmas, count in sample_lemma_counts.items():
    for src, tgt, src_msd, tgt_msd, _ in corrects:
        # We get the lemma/POS combo shared between the src and target
        # TODO: Remove lower() once casing is resolved
        # lemma_poses = set(lex[src.lower()]) & set(lex[tgt.lower()])
        lemma_poses = set(lex[src]) & set(lex[tgt])
        sigmorphon_samples = [
            sample for lemma_pos in lemma_poses for sample in sigmorphon_dict.get(lemma_pos, [])
        ]
        sig_count = len(sigmorphon_samples)
        if sig_count > 0:
            new_samples = random.sample(sigmorphon_samples, min(1, sig_count))
            _update_lemma_counts(new_samples)
            # Add the other samples to the MSD dict for the second round of sampling
            _update_sig_msd_dict([s for s in sigmorphon_samples if s not in new_samples])
            samples.extend(new_samples)

    print(
        f"Found {sum(sigmorphon_lemma_counts.values())} replacement samples with lemmas overlapping the corrects"
    )

    for msd, count in sample_msd_counts.items():
        sig_count = len(sigmorphon_msd_dict.get(msd, []))
        requested = count - len(sigmorphon_msd_dict.get(msd, []))

        if requested > 0 and sig_count > 0:
            new_samples = random.sample(
                sigmorphon_msd_dict[msd],
                min(requested, len(sigmorphon_msd_dict[msd]))
            )

            samples.extend(new_samples)

    leftovers = [s for s in leftovers if s not in samples]
    return samples, leftovers


def read_test_lemmas(fn: str) -> Set:
    lemmas = set()
    with open(fn, "r") as f:
        for line in f:
            lemma, tgt, msd = line.rstrip().split("\t")
            lemmas.add(lemma)

    return lemmas


def sigmorphon2noisy_format(sig_sample: Tuple) -> Tuple:
    src, tgt, tgt_msd = sig_sample
    # This might not really make sense for non-verbs, but since we ignore
    # src_msd in practice, it shouldnt matter
    src_msd = tgt_msd.split(";")[0] + ";NFIN"
    annotation = "C"

    return (src, tgt, src_msd, tgt_msd, annotation)


def make_resampled(samples_list: List, new_samples: List) -> List:
    MAX_CORRECT_IDX = max([i for i, data in enumerate(samples_list) if data[-1] == "C"])
    resampled = samples_list.copy()
    copy_new_samples = new_samples.copy()
    print(f"Resampling from {len(copy_new_samples)} SIGMORPHON samples")
    for i, sample in enumerate(resampled):
        if len(copy_new_samples) < 1 and i < MAX_CORRECT_IDX:
            print("WARNING: Ran out of new samples before replacing all corrects!")
            break
        if sample[-1] == "C":
            resampled[i] = sigmorphon2noisy_format(copy_new_samples.pop(0))

    print(f"Replaced corrects with new samples. {len(copy_new_samples)} unused new samples remain.")
    return resampled


def compute_apertium_seen_unseen(
    samples: List[str],
    apertium_lexicon: Dict,
    test_lemma_set: Set
):
    num_seen = 0
    num_unseen = 0
    # Temp lower for matching apertium
    # TODO: Remove lower() once casing is resolved
    # test_lemma_set = set([l.lower() for l in test_lemma_set])
    test_lemma_set = set([l for l in test_lemma_set])
    for src, tgt, src_msd, tgt_msd, _ in samples:
        seen = False
        # We get the lemma/POS combo shared between the src and target
        # TODO: Remove lower() once casing is resolved
        # lemma_poses = set(apertium_lexicon[src.lower()]) & set(apertium_lexicon[tgt.lower()])
        lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
        for lemma, pos in lemma_poses:
            if lemma in test_lemma_set:
                seen = True
                # Each word can only count once towards seen/unseen
                break

        if seen:
            num_seen += 1
        else:
            num_unseen += 1

    return num_seen, num_unseen


def summarize_results(
    samples_list: List,
    resampled: List,
    test_lemma_set: Set,
    apertium_lexicon: Dict
):
    print("\nSummarizing dataset modifications after resampling:")
    old_msd_counts = get_samples_MSD_counts(samples_list)
    # old_lemma_counts = get_samples_lemma_counts(samples_list, apertium_lexicon)

    new_msd_counts = get_samples_MSD_counts(resampled)
    # The corrects in resampled are from SIGMORPHON, so not always in apertium_lex
    # and the src word is the lemma anyway.
    new_lemmas = set()
    for src, tgt, src_msd, tgt_msd, ann in resampled:
        if ann == "C":
            new_lemmas.add(src)
        else:
            # TODO: Remove lower() once casing is resolved
            # lemma_poses = set(apertium_lexicon[src.lower()]) & set(apertium_lexicon[tgt.lower()])
            lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
            for lemma, pos in lemma_poses:
                new_lemmas.add(lemma)

    overlapping_lemma_cnt = 0
    for src, tgt, src_msd, tgt_msd, ann in samples_list:
        if ann == "C":
            # We get the lemma/POS combo shared between the src and target
            # TODO: Remove lower() once casing is resolved
            # lemma_poses = set(apertium_lexicon[src.lower()]) & set(apertium_lexicon[tgt.lower()])
            lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
            for lemma, pos in lemma_poses:
                if lemma in new_lemmas:
                    overlapping_lemma_cnt += 1
                    # Each word can only count once towards seen/unseen
                    break
    
    msg_percent = round(overlapping_lemma_cnt / len(samples_list) * 100, 2)
    print(
        f"Resampled dataset lemma overlap with old: {msg_percent}%"
    )

    overlapping_msd_cnt = 0
    for msd in set(old_msd_counts.keys()).union(set(new_msd_counts.keys())):
        old_cnt = old_msd_counts.get(msd, 0)
        new_cnt = new_msd_counts.get(msd, 0)
        # The min of counts for a given msd is how many overlap.
        overlapping_msd_cnt += min(old_cnt, new_cnt)

    msg_percent = round(overlapping_msd_cnt / len(samples_list) * 100, 2)
    print(
        f"Resampled dataset MSD overlap with old: {msg_percent}%"
    )

    num_old_seen, num_old_unseen = compute_apertium_seen_unseen(
        samples_list, apertium_lexicon, test_lemma_set
    )

    new_corrects = [s for s in resampled if s[-1] == "C"]
    new_incorrects = [s for s in resampled if s[-1] != "C"]
    num_new_seen, num_new_unseen = compute_apertium_seen_unseen(
        new_incorrects, apertium_lexicon, test_lemma_set
    )
    for s in new_corrects:
        if s[0] in test_lemma_set:
            num_new_seen += 1
        else:
            num_new_unseen += 1

    print(f"old seen/unseen: {num_old_seen}/{num_old_unseen}")
    print(f"new seen/unseen: {num_new_seen}/{num_new_unseen}")


def make_lemmas_set(samples_list, apertium_lexicon, sigmorphon=False):
    lemmas = set()
    for src, tgt, *tail in samples_list:
        if len(tail) == 1:
            print(tail)
            lemmas.add(src)
        elif sigmorphon and tail[-1] == "C":
            lemmas.add(src)
        else:
            lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
            for lemma, pos in lemma_poses:
                lemmas.add(lemma)

    return lemmas


@click.command()
@click.option("--samples_path", required=True)
@click.option("--sigmorphon_path", required=True)
@click.option("--test_path", required=True)
@click.option("--apertium_lexicon_path", required=True)
@click.option("--output_path", required=True)
def main(samples_path, sigmorphon_path, test_path, apertium_lexicon_path, output_path):
    # 1. Read everything
    samples_list = read_samples(samples_path)
    # 2. Get corrects samples
    corrects = [s for s in samples_list if s[-1] == "C"]
    print(f"Found {len(corrects)} correct samples to replace from SIGMORPHON training")
    # 3. Get the samples lemmas and MSDs counts
    corrects_msd_counts = get_samples_MSD_counts(corrects)
    original_msds = set(corrects_msd_counts.keys())
    # Get SIGMORPHON data in a dict hashed by lemma
    # Ignore forms with MSDs not in the original data
    sigmorphon_dict = read_sigmorphon(sigmorphon_path, original_msds)
    test_lemma_set = read_test_lemmas(test_path)
    apertium_lexicon = read_apertium_lexicon(apertium_lexicon_path)
    
    assert(sum(corrects_msd_counts.values()) == len(corrects))
    
    # 4. Sample as many of the same lemma and MSD from SIGMORPHON
    #    as possible (but not more than in the counts dicts found above)
    print("Performing initial sampling to match lemmas and MSDs as much as possible")
    new_samples, leftover_sigmorphon = sample_sigmorphon_matches(
        sigmorphon_dict,
        corrects,
        apertium_lexicon,
        corrects_msd_counts
    )

    if new_samples == len(corrects):
        # Then we are done.
        print("Initial sampling got new samples to replace all corrects!")
        resampled = make_resampled(samples_list, new_samples)
    else:
        print(f"Initial sampling complete ({len(new_samples)}). Sampling more seen/unseen lemmas ")
        print(f"from leftover SIGMORPHON ({len(leftover_sigmorphon)})")
        # 5. Compute seen/unseen lemma counts for original samples
        num_corrects_seen, num_corrects_unseen = compute_apertium_seen_unseen(
            corrects, apertium_lexicon, test_lemma_set
        )
        print(f"Corrects seen/unseen: {num_corrects_seen}//{num_corrects_unseen}")

        # 6. Split SIGMORPHON into seen/unseen in terms of test data lemmas
        sig_seen = [s for s in leftover_sigmorphon if s[0] in test_lemma_set]
        sig_unseen = [s for s in leftover_sigmorphon if s[0] not in test_lemma_set]
        # 7. For the rest of the SIGMORPHON data, sample (without replacement) from seen/unseen
        #    to get the same seen/unseen counts as in the original samples
        # New samples are lemma, form, msd
        new_seens = [
            s for s in new_samples if s[0] in test_lemma_set
        ]
        new_unseens = [
            s for s in new_samples if s[0] not in test_lemma_set
        ]
        print(f"Number of new samples with seen lemma:{len(new_seens)}")
        print(f"Number of new samples with unseen lemma:{len(new_unseens)}")
        requested_seen = num_corrects_seen - len(new_seens)
        new_seen_lemma_samples = []
        if requested_seen > 0:
            print(f"Requesting {requested_seen} seen lemmas from sigmorphon ({len(sig_seen)})")
            random.shuffle(sig_seen)
            new_seen_lemma_samples = sig_seen[:min(requested_seen, len(sig_seen))]
            sig_seen = sig_seen[min(requested_seen, len(sig_seen)):]
            # new_seen_lemma_samples = random.sample(
            #     sig_seen,
            #     min(requested_seen, len(sig_seen))
            # )
        elif requested_seen == 0:
            msg = f"No need to request more seen lemmas, "
            msg += f"we already have {len(new_seens)} == {num_corrects_seen}"
            print(msg)
        else:
            new_seen_removes = len(new_seens) - num_corrects_seen
            msg = f"Redistributing {new_seen_removes} samples "
            msg += "from seen lemmas to sample unseen lemmas."
            new_seens = new_seens[:num_corrects_seen]
            msg += f"Resulting in {len(new_seens)} new seen lemmas"
            print(msg)

        requested_unseen = num_corrects_unseen - len(new_unseens)
        new_unseen_lemma_samples = []
        if requested_unseen > 0:
            print(f"Requesting {requested_unseen} unseen lemmas from sigmorphon ({len(sig_unseen)})")
            random.shuffle(sig_unseen)
            new_unseen_lemma_samples = sig_unseen[:min(requested_unseen, len(sig_unseen))]
            sig_unseen = sig_unseen[min(requested_unseen, len(sig_unseen)):]
            # new_unseen_lemma_samples = random.sample(
            #     sig_unseen,
            #     min(requested_unseen, len(sig_unseen))
            # )
        elif requested_unseen == 0:
            msg = f"No need to request more unseen lemmas, "
            msg += f"we already have {len(new_unseens)} == {num_corrects_unseen}"
            print(msg) 
        else:
            new_unseen_removes = len(new_unseens) - num_corrects_unseen
            msg = f"Redistributing {new_unseen_removes} samples "
            msg += "from unseen lemmas to sample seen lemmas."
            new_unseens = new_unseens[:num_corrects_unseen]
            msg += f"Resulting in {len(new_unseens)} new unseen lemmas"
        
        new_samples = new_seens + new_unseens
        random.shuffle(new_samples)
        seen_unseen = new_seen_lemma_samples + new_unseen_lemma_samples
        random.shuffle(seen_unseen)
        resampled = make_resampled(
            samples_list,
            new_samples + seen_unseen
        )

    # Print SIGMORPHON train set overlap just for fun
    train_lemmas = set([lemma for samples in sigmorphon_dict.values() for lemma, _, _ in samples])
    test_seen = len(train_lemmas & test_lemma_set)
    msg_percent = round(test_seen/len(test_lemma_set) * 100, 2)
    print(
        f"{test_seen} ({msg_percent})% SIGMOPRPHON dev are seen in original SIGMOPRPHON train"
    )

    summarize_results(samples_list, resampled, test_lemma_set, apertium_lexicon)

    all_old_lemmas = make_lemmas_set(samples_list, apertium_lexicon)
    all_new_lemmas = make_lemmas_set(resampled, apertium_lexicon, sigmorphon=True)

    # I realized that the above resampling focuses on tokens with a lemma in test, not types.
    # Here we do one more resampling to fix for types.
    # TODO: Eventually go back and just sample for types in the 
    # first place to clean this script up...
    # lower_test_lemma_set = set([l.lower() for l in test_lemma_set])
    old_overlapping_lemma_types = test_lemma_set & all_old_lemmas
    new_overlapping_lemma_types = test_lemma_set & all_new_lemmas
    overlap_diff = len(new_overlapping_lemma_types) - len(old_overlapping_lemma_types)
    new_sigmorphon_overlapping_lemmas = new_overlapping_lemma_types - old_overlapping_lemma_types
    if overlap_diff > 0:
        # Then remove some new lemmas
        removes = list(new_sigmorphon_overlapping_lemmas)[:overlap_diff]

        print(f"{overlap_diff} too many overlapping lemma types in resampled data, replacing some...")
        print(f"Removing samples with lemmas in {removes}")
        leftover_unseens = sig_unseen
        for i, sample in enumerate(resampled):
            src, tgt, src_msd, tgt_msd, ann = sample
            if ann == "C":
                if src in removes:
                    while len(leftover_unseens) > 0:
                        leftover_unseen = leftover_unseens.pop(0)
                        if leftover_unseen[0] not in test_lemma_set:
                            resampled[i] = sigmorphon2noisy_format(leftover_unseen)
                            break
            # else:
            #     lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
            #     for lemma, pos in lemma_poses:
            #         if lemma in removes:
            #             while len(leftover_unseens) > 0:
            #                 leftover_unseen = leftover_unseens.pop(0)
            #                 if leftover_unseen[0] not in all_new_lemmas:
            #                     resampled[i] = sigmorphon2noisy_format(leftover_unseen)
            #                     break
            #             continue
    elif overlap_diff < 0:
        # Then we need more seen lemmas
        print(f"{overlap_diff} Not enough overlapping lemma types in resampled data, replacing some...")
        # Then remove some new lemmas
        leftover_seens = sig_seen
        for i, sample in enumerate(resampled):
            if overlap_diff >= 0 :
                break
            src, tgt, src_msd, tgt_msd, ann = sample
            if ann == "C":
                if src not in test_lemma_set:
                    while len(leftover_seens) > 0:
                        leftover_seen = leftover_seens.pop(0)
                        if leftover_seen[0] not in all_new_lemmas:
                            resampled[i] = sigmorphon2noisy_format(leftover_seen)
                            overlap_diff += 1
                            break
            # else:
            #     lemma_poses = set(apertium_lexicon[src]) & set(apertium_lexicon[tgt])
            #     for lemma, pos in lemma_poses:
            #         if lemma not in all_new_lemmas:
            #             while len(leftover_seens) > 0:
            #                 leftover_seen = leftover_seens.pop(0)
            #                 if leftover_seen[0] not in all_new_lemmas:
            #                     resampled[i] = leftover_seen
            #                     break
            #             continue

    all_old_lemmas = make_lemmas_set(samples_list, apertium_lexicon)
    all_new_lemmas = make_lemmas_set(resampled, apertium_lexicon, sigmorphon=True)
    old_overlapping_lemma_types = test_lemma_set & all_old_lemmas
    new_overlapping_lemma_types = test_lemma_set & all_new_lemmas
    msg_percent = len(old_overlapping_lemma_types) / len(test_lemma_set)
    print(
        f"Previous test set lemma coverage: {len(old_overlapping_lemma_types)} ({round(msg_percent*100, 2)})%"
    )
    msg_percent = len(new_overlapping_lemma_types) / len(test_lemma_set)
    print(
        f"New test set lemma coverage: {len(new_overlapping_lemma_types)} ({round(msg_percent*100, 2)})%"
    )

    print(f"\nWriting to {output_path}")
    with open(output_path, "w") as out:
        for sample in resampled:
            out.write("\t".join(sample))
            out.write("\n")


if __name__ == "__main__":
    main()