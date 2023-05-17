"""
Written 10/24/2022

New script for resampling reinflection from SIGMORPHON.

We build reinflection samples from UniMorph from the MSDs in the oriinal tUMPC dataset.
From that set, we sample downsample to those lemmas in tUMPC. Then, we sample from the set of lemmas + other lemmas if we need them to get enough data, controlling for word length.

We do this by building a distribution over (tgt) word lengths and sampling from that.

The intuition is that UniMorph data tends to be longer words on avg apparently? And we 
want a dataset that is as similar as possible to tUMPC data, other than potentially containing
a larger diversity of inflection classes."""


import click
import itertools
from collections import Counter
from typing import Dict, List, Tuple
import random

from tqdm import tqdm


def read_tUMPC(filename):
    samples = []
    with open(filename, "r") as f:
        for line in f:
            src, tgt, src_msd, tgt_msd, anns = line.rstrip().split("\t")
            samples.append((src, tgt, src_msd, tgt_msd, anns))

    return samples


def read_unimorph(filename):
    paradigms = {}
    with open(filename, "r") as f:
        for line in f:
            if line.rstrip():
                lemma, form, msd = line.rstrip().split("\t")
                paradigms.setdefault(lemma, []).append([form, msd])

    lemma2sample = {}
    for lemma, forms in paradigms.items():
        pairs = itertools.permutations(forms, r=2)
        # src, tgt, src_msd, tgt_msd, annotation
        pairs = [(x[0], y[0], x[1], y[1], "C") for x, y in pairs]
        lemma2sample[lemma] = pairs
        
    uni_lemmatizer = {}
    for lemma, samples in lemma2sample.items():
        for s in samples:
            uni_lemmatizer[s[0]] = lemma
    
    return lemma2sample, uni_lemmatizer


def read_test_lemmas_and_msds(test_filepath):
    lemmas = set()
    msds = set()
    pairs = set()
    with open(test_filepath, "r") as f:
        for line in f:
            lemma, form, msd = line.rstrip().split("\t")
            lemmas.add(lemma)
            msds.add(msd)
            pairs.add((form, msd))

    return lemmas, msds, pairs


def get_tumpC_MSDS(data):
    return set([tgt_msd for _, _, _, tgt_msd, _ in data])


# POS_MAP = {
#     "vblex": "V",
#     "vaux": "V",
#     "vbmod": "V",
#     "vbser": "V",
#     "vbhaver": "V",
#     "n": "N",
#     "np": "PROPN",
#     "adj": "ADJ",
#     "adv": "ADV",
#     "prn": "PRN",
#     "num": "NUM",
#     "det": "DET",
#     "pr": "PREP"
# }

# def map_apert_pos(apert_pos: str) -> str:
#     out = POS_MAP.get(apert_pos)

#     if out == None:
#         return apert_pos

#     return out


def make_tumpc2word2lemma(filepath):
    lexicon = {}
    with open(filepath, "r") as f:
        for line in f:
            word, analyses_str = line.rstrip().split("\t")
            analyses = analyses_str.split(",")
            for analysis in analyses:
                fields = analysis.split(";")
                lemma = fields[0]

                lexicon.setdefault(word, []).append(lemma)

    return lexicon


def get_tUMPC_word_length_counts(tgt_words):
    return Counter([len(w) for w in tgt_words])


def make_length2pairs(uni_data, threshold=None):
    len2samples = {}
    tgts_cnt = {}

    random.shuffle(uni_data)
    for src, tgt, src_msd, tgt_msd, c in uni_data:
        if threshold is None or tgts_cnt.get(tgt, 0) < threshold:
            tgts_cnt.setdefault(tgt, 0)
            tgts_cnt[tgt] += 1
            len2samples.setdefault(len(tgt), []).append((src, tgt, src_msd, tgt_msd, c))

    return len2samples


def summarize(
    samples: List[Tuple],
    old_samples: List[Tuple],
    test_lemmas: List[str],
    test_msds: List[str],
    tumpc_lemmas: List[str],
    tumpc_msds: List[str],
    uni_lemmatizer: Dict[str, str]
):
    sample_lemmas = set([uni_lemmatizer[s[0]] for s in samples])
    sample_msds = set([s[3] for s in samples])
    
    test_tumpc_lemma_overlap = len(test_lemmas & tumpc_lemmas) / len(test_lemmas)
    msg = f"OLD tUMPC Correct lemma overlap with test set: {round(test_tumpc_lemma_overlap, 2) * 100}%"
    print(msg)

    test_tumpc_msd_overlap = len(tumpc_msds & test_msds) / len(test_msds)
    msg = f"OLD tUMPC Correct msd overlap with test set: {round(test_tumpc_msd_overlap, 2) * 100}%"
    print(msg)

    leng_counts = get_tUMPC_word_length_counts([s[1] for s in old_samples])
    msg = f"OLD tUMPC word length counts:"
    for k, v in leng_counts.items():
        msg += f"\n{k}: {v}"
    print(msg)

    tumpc_lemma_overlap = len(sample_lemmas & tumpc_lemmas) / len(tumpc_lemmas)
    msg = f"New Correct lemma overlap with tUMPC corrects: {round(tumpc_lemma_overlap, 2) * 100}%"
    print(msg)

    tumpc_msd_overlap = len(sample_msds & tumpc_msds) / len(tumpc_msds)
    msg = f"New Correct msd overlap with tUMPC corrects: {round(tumpc_msd_overlap, 2) * 100}%"
    print(msg)

    test_lemma_overlap = len(sample_lemmas & test_lemmas) / len(test_lemmas)
    msg = f"New Correct lemma overlap with test set: {round(test_lemma_overlap, 2) * 100}%"
    print(msg)

    test_msd_overlap = len(sample_msds & test_msds) / len(test_msds)
    msg = f"New Correct msd overlap with test set: {round(test_msd_overlap, 2) * 100}%"
    print(msg)

    leng_counts = get_tUMPC_word_length_counts([s[1] for s in samples])
    msg = f"New word length counts:"
    for k, v in leng_counts.items():
        msg += f"\n{k}: {v}"
    print(msg)

    msg = f"\nOld MSD counts:"
    sample_msds = Counter([s[3] for s in old_samples])
    for k, v in sorted(sample_msds.items()):
        msg += f"\n{k}: {v}"
    print(msg)

    msg = f"\nNEW MSD counts:"
    sample_msds = Counter([s[3] for s in samples])
    for k, v in sorted(sample_msds.items()):
        msg += f"\n{k}: {v}"
    print(msg)


LANG2NEW_LEMMA_RATIO = {
    "deu": 15,
    "swe": 30,
    "isl": 45,
    "rus": 45,
}

@click.command()
@click.option("--tumpc-filepath", required=True)
@click.option("--lang", required=True)
@click.option("--apt-filepath", required=True)
@click.option("--unimorph-filepath", required=True)
@click.option("--test-filepath", required=True)
@click.option("--output-filepath", required=True)
def main(tumpc_filepath, lang, apt_filepath, unimorph_filepath, test_filepath, output_filepath):
    # print(len([1,2,3]))
    # 1. Read tUMPC, and its MSDs and Lemmas
    print("Reading tUMPC and stats")
    tumpc_samples = read_tUMPC(tumpc_filepath)
    tumpc_w2lemma = make_tumpc2word2lemma(apt_filepath)
    tumpc_lemmas = set([l for k, lemmas in tumpc_w2lemma.items() for l in lemmas])
    tumpc_corrects = [s for s in tumpc_samples if s[-1] == "C"]
    # 2. Read UniMorph
    print("Reading UniMorph inflection pairs")
    uni_lemma2samp, uni_lemmatizer = read_unimorph(unimorph_filepath)
    uni_samples = [sample for samples in uni_lemma2samp.values() for sample in samples]
    # 3. Get unimorph of only tUMPC MSDs
    print("Filtering uni pairs to only those with MSDs in the set of tUMPC correct data")
    tumpc_msds = get_tumpC_MSDS(tumpc_samples)
    tumpc_correct_msds = get_tumpC_MSDS(tumpc_corrects)
    tumpc_correct_lemmas = set([
        l for src, _, _, _, ann in tumpc_samples for l in tumpc_w2lemma[src] if ann == "C"
    ])
    # s[3] is the tgt msd
    uni_valid_msd_pairs = [s for s in uni_samples if s[3] in tumpc_correct_msds]
    uni_ALL_len2pairs = make_length2pairs(uni_valid_msd_pairs)
    # 4. Get subsample of 3) with only tUMPC lemmas
    print("Second filtering stage: keeping only those with lemmas in tUMPC")
    num_overlapping_lemmas = len(set(uni_lemma2samp.keys()) & tumpc_correct_lemmas)
    print(f"\n {num_overlapping_lemmas} out of {len(uni_lemma2samp)} UniMorph lemmas overlap with tUMPC correct data.")
    print(f"This comprises {len([v for vals in uni_lemma2samp.values() for v in vals])} pairs.")
    uni_valid_lemma_msd_pairs = [
        s for s in uni_valid_msd_pairs if uni_lemmatizer[s[0]] in tumpc_correct_lemmas
    ]
    print(f"Got {len(uni_valid_lemma_msd_pairs)} UniMorph samples with both lemma AND msd overlapping tUMPC corrects")
    print("\nSAMPLING from additional lemmas to increase diversity.")
    uni_invalid_lemmas = [
        l for l in uni_lemma2samp.keys() if l not in tumpc_correct_lemmas
    ]
    uni_invalid_lemmas = random.sample(
        uni_invalid_lemmas, 
        int(num_overlapping_lemmas / LANG2NEW_LEMMA_RATIO[lang])
    )
    uni_invalid_lemma_valid_msd_pairs = [
        s for s in uni_valid_msd_pairs if uni_lemmatizer[s[0]] in uni_invalid_lemmas
    ]
    print(f"Adding {len(uni_invalid_lemmas)} new lemmas comprising {len(uni_invalid_lemma_valid_msd_pairs)} pairs.")
    test_lemmas, test_msds, test_pairs = read_test_lemmas_and_msds(test_filepath)
    # 5. Get tUMPC word-len distribution
    print("Computing word_length distribution")
    length_counts = get_tUMPC_word_length_counts([tgt for _, tgt, _, _, _ in tumpc_corrects])
    print("Found for tUMPC:")
    for k, v in length_counts.items():
        print(f"{k}: {v}")
    # 6. Sample from valid UniMorph pairs according to that dist.
    final_uni = uni_valid_lemma_msd_pairs + uni_invalid_lemma_valid_msd_pairs
    # random.sample(
    #     uni_invalid_lemma_valid_msd_pairs,
    #     new_lemma_sample_size
    # )
    uni_length2pairs = make_length2pairs(final_uni, threshold = 10)

    print("Found for UniMorph:")
    for k, v in uni_length2pairs.items():
        print(f"{k}: {len(v)}")
    # 7. Sample more lemmas according to the lemma overlap dist until
    #     we have enough data matching the word length dist.
    tumpc_overlapping_lemma_cnt = len(tumpc_correct_lemmas & test_lemmas)
    tumpc_overlapping_lemma_pct = tumpc_overlapping_lemma_cnt / len(test_lemmas)
    # Get num lemmas overlapping in the unimorph set to compare to tUMPC num
    uni_lemmas = set([uni_lemmatizer[src] for src, _, _, _, _ in uni_samples])
    uni_overlapping_lemma_cnt = len(uni_lemmas & test_lemmas)
    uni_overlapping_lemma_pct = uni_overlapping_lemma_cnt / len(test_lemmas)
    print(f"Overlapping lemmas in tUMPC CORRECT data: {round(tumpc_overlapping_lemma_pct, 2)*100}%")
    print(f"Overlapping lemmas in UniMorph: {round(uni_overlapping_lemma_pct, 2) * 100}%")
    print()
    tumpc_overlapping_msd_cnt = len(tumpc_correct_msds & test_msds)
    tumpc_overlapping_msd_pct = tumpc_overlapping_msd_cnt / len(test_msds)
    # Get num msds overlapping in the unimorph set to compare to tUMPC num
    uni_msds = set([msd for _, _, _, msd, _ in uni_samples])
    uni_overlapping_msd_cnt = len(uni_msds & test_msds)
    uni_overlapping_msd_pct = uni_overlapping_msd_cnt / len(test_msds)
    print(f"Overlapping MSDs in tUMPC CORRECT data: {round(tumpc_overlapping_msd_pct, 2)*100}%")
    print(f"Overlapping MSDs in UniMorph: {round(uni_overlapping_msd_pct, 2)*100}%")
    uni_leftover_len2pairs = {}
    print("Getting valid UniMorph pairs")
    uni_valid_lemma_msd_pairs = set(uni_valid_lemma_msd_pairs)
    for l, pairs in tqdm(uni_ALL_len2pairs.items()):
        uni_leftover_len2pairs[l] = [s for s in pairs if s not in uni_valid_lemma_msd_pairs]
            
    resampled_corrects = []
    print("Sampling from the same lengths...")
    for length, cnt in length_counts.items():
        uni = uni_length2pairs[length]
        if len(uni) < cnt:
            print(f"upsampling {cnt - len(uni)} for word length {length}...")
            # Upsample
            # get from NOT test lemmas if overlap is the same/too high
            if uni_overlapping_lemma_cnt < tumpc_overlapping_lemma_cnt:
                add_ins = [s for s in uni_leftover_len2pairs[length] if s[0] not in test_lemmas]
                uni += add_ins[:cnt - len(uni)]
            # get from seen test lemmas otherwise
            else:
                add_ins = [s for s in uni_leftover_len2pairs[length] if s[0] in test_lemmas]
                add_ins = add_ins[:tumpc_overlapping_lemma_cnt - uni_overlapping_lemma_cnt]
                add_ins += [s for s in uni_leftover_len2pairs[length] if s[0] not in test_lemmas]
                uni += add_ins[:cnt - len(uni)]
        
        # Sample double the tUMPC count so we have the distribution, with extra data to sample from.
        # cnt = len(uni)

        # if length == 14:
        #     print(set([p[1] for p in uni]))
        #     print(uni)
        print(f"sampling {cnt} pairs of length {length} from UniMorph. {len(set([p[1] for p in uni]))} unique targets of {len(uni)} pairs")
        resampled_corrects.extend(random.sample(uni, cnt))

    random.shuffle(resampled_corrects)
    num_resampled = len(resampled_corrects)
    print(f"Sampled {num_resampled} new corrects")
    print(f"Writing to {output_filepath} with corrects replaced by unimorph reinfl samples")
    written_samples = []
    with open(output_filepath, "w") as out:
        for src, tgt, src_msd, tgt_msd, anns in tumpc_samples:
            if anns != "C":
                print("\t".join([src, tgt, src_msd, tgt_msd, anns]), file=out)
            else:
                sample = resampled_corrects.pop(0)
                written_samples.append(sample)
                print("\t".join(sample), file=out)

    summarize(
        written_samples,
        tumpc_corrects,
        test_lemmas,
        test_msds,
        tumpc_correct_lemmas,
        tumpc_correct_msds,
        uni_lemmatizer
    )
    print(f"Wrote {num_resampled - len(resampled_corrects)} newly sampled corrects, and all noise")


if __name__ == "__main__":
    main()
