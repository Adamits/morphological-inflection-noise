from collections import Counter
import numpy as np
from scipy import stats

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


def read_apertium_lexicon(fn: str):
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


def read_data(fn):
    samples = []
    with open(fn, "r") as f:
        for line in f:
            if line.rstrip():
                samples.append(line.rstrip().split("\t"))

    return samples


def make_lemmas_set(samples_list, apertium_lexicon):
    lemmas = set()
    for src, tgt, *tail in samples_list:
        if len(tail) == 1:
            lemmas.add(src)
        else:
            lemma_poses = set(apertium_lexicon.get(src, set())) & set(apertium_lexicon.get(tgt, set()))
            # If no apt analysis, it must be from UniMorph.
            if len(lemma_poses) < 1:
                lemmas.add(src)
            else:
                for lemma, pos in lemma_poses:
                    lemmas.add(lemma)

    return lemmas


def get_type_token_ratio(data):
    types = set()
    num_tokens = 0

    tgt_types = set()
    for src, tgt, *tail in data:
        types.add(src)
        types.add(tgt)
        tgt_types.add(tgt)
        num_tokens += 2
    
    ttr = len(types) / num_tokens
    tgt_ttr = len(tgt_types) / (num_tokens / 2)
    return ttr, tgt_ttr


def get_MSD_entropy(data):
    msds = [msd for _, _, _, msd, _ in data]
    value,counts = np.unique(
        msds, return_counts=True
    )
    return stats.entropy(counts)


def get_msds(samples_list):
    msds = set()
    for samp in samples_list:
        if len(samp) == 3:
            _, _, msd = samp
            msds.add(msd)
        elif len(samp) == 5:
            _, _, _, msd, _ = samp
            msds.add(msd)
        else:
            raise Exception(f"Sample length {len(samp)}?")

    return msds


def get_tags(msds):
    return set([t for msd in msds for t in msd.split(";")])


def main(train_fn, test_fn, apertium_lexicon_path, lang):
    stats_dict = {"Lang": lang}
    apertium_lexicon = read_apertium_lexicon(apertium_lexicon_path)
    train_data = read_data(train_fn)
    stats_dict["Num Train Samples"] = str(len(train_data))
    # print(f"Number of training samples: {len(train_data)}")
    test_data = read_data(test_fn)

    test_lemmas = set([lem for lem, surf, msd in test_data])
    train_lemmas = make_lemmas_set(train_data, apertium_lexicon)
    # print(train_lemmas & test_lemmas)
    test_seen = len(test_lemmas & train_lemmas)
    msg_percent = round(test_seen/len(test_lemmas) * 100, 2)
    # print(f"LEMMAS OVERLAPPING WITH TEST: {test_seen}, {len(test_lemmas)} ({msg_percent}%)")
    stats_dict["Test lemma overlap"] = str(msg_percent)

    train_msds = get_msds(train_data)
    test_msds = get_msds(test_data)
    test_seen_msds = len(train_msds & test_msds)
    msg_percent = round(test_seen_msds/len(test_msds) * 100, 2)
    # print(f"MSDS OVERLAPPING WITH TEST: {test_seen_msds}, {len(test_msds)} ({msg_percent}%)")
    stats_dict["Test MSD overlap"] = str(msg_percent)

    train_tags = get_tags(train_msds)
    test_tags = get_tags(test_msds)
    test_seen_tags = len(train_tags & test_tags)
    msg_percent = round(test_seen_tags/len(test_tags) * 100, 2)
    stats_dict["Test tag overlap"] = str(msg_percent)

    # print(f"Number unique lemmas: {len(train_lemmas)}")
    stats_dict["unique lemmas"] = str(len(train_lemmas))
    # print(f"Number unique MSDs: {len(train_msds)}")
    stats_dict["unique MSDs"] = str(len(train_msds))
    ttr, tgt_ttr = get_type_token_ratio(train_data)
    # print(f"Type token ratio: {round(ttr, 2)}")
    stats_dict["TTR"] = str(round(ttr, 2))
    # print(f"Type token ratio for target tokens only: {round(tgt_ttr, 2)}")
    stats_dict["Target TTR"] = str(round(tgt_ttr, 2))

    MSD_entropy = get_MSD_entropy(train_data)
    # print(f"MSD entropy: {round(MSD_entropy, 2)}")
    stats_dict["MSD entropy"] = str(round(MSD_entropy, 2))

    tgt2msds = {}
    for _, tgt, _, msd, _ in train_data:
        tgt2msds.setdefault(tgt, set()).add(msd)
    # print(f"Average # of MSDs per tgt form: {sum([len(v) for v in tgt2msds.values()]) / len(tgt2msds)}")
    stats_dict["MSDs per tgt"] = str(round(sum([len(v) for v in tgt2msds.values()]) / len(tgt2msds), 4))

    tgt_token_counts = Counter([tgt for src, tgt, *tail in train_data])
    for k in [5, 10, 15, 20]:
        num_dups_gtr_k = len([t for t, cnt in tgt_token_counts.items() if cnt >= k])
        # print(f"Number of target types with >= {k} instances: {num_dups_gtr_k}")
        stats_dict[f"Num type freqs > {k}"] = str(num_dups_gtr_k)
    
    frequent_tgt = sorted(tgt_token_counts.items(), key=lambda x: x[1], reverse=True)[0]
    stats_dict[f"Most frequent tgt"] = str(frequent_tgt)

    # print("\t".join(stats_dict.keys()))
    # print("\t".join(stats_dict.values()))
    return stats_dict


if __name__ == "__main__":
    stats_dicts = []
    for language, sig_language in [
        ("deu", "german"),
        ("swe", "swedish"),
        ("isl", "icelandic"),
        ("rus", "russian")
    ]:
        # train_fn = "inflection/data/train/sig2017/deu-train-low"
        # train_fn = "noise-annotator/data/sampled/corrects/deu_sampled.tsv"
        # train_fn = "noise-annotator/data/sampled/corrects/deu_sampled_sigmorphon_resampling.tsv"

        # f"inflection/data/train/sig2017/{language}-train-low",
        # f"inflection/data/train/sig2017/{language}-train-medium",
        # f"../sigmorphon/sigmorphon-data/data/{sig_language}-train-high",
        # for train_fn in [
        #     f"noise-annotator/data/sampled/baseline/{language}_sampled.tsv",
        #     f"noise-annotator/data/sampled/baseline/msd_sampled/{language}_sampled_sigmorphon_resampling_reinflection.tsv",
        #     f"noise-annotator/data/sampled/baseline/msd_sampled/{language}_sampled_sigmorphon_resampling.tsv"
        # ]:
        #     test_fn = f"../sigmorphon/sigmorphon-data/data/{sig_language}-dev"
        #     test_fn = f"inflection/data/test/{sig_language}-dev"
        #     apertium_lexicon_fn = f"noise-annotator/data/tUMPC/lexicon/{language}_apertium.tsv"
        #     print(train_fn)
        #     main(train_fn, test_fn, apertium_lexicon_fn)
        #     print("="*100)
        # train_fn = f"noise-annotator/data/sampled/baseline/msd_sampled/{language}_sampled_sigmorphon_resampling_reinflection.tsv"
        # train_fn = f"noise-annotator/data/sampled/corrects/{language}_sampled.tsv"
        # train_fn = f"noise-annotator/data/sampled/corrects/length_controlled/{language}_sampled_sigmorphon_resampling_reinflection.tsv"
        # train_fn = f"noise-annotator/data/sampled/corrects/TEST_length_controlled/{language}_sampled_sigmorphon_resampling_reinflection.tsv"
        # train_fn = f"noise-annotator/data/sampled/baseline/{language}_sampled.tsv"
        train_fn = f"noise-annotator/data/sampled/baseline/{language}_sampled.tsv"
        test_fn = f"../sigmorphon/sigmorphon-data/data/{sig_language}-dev"
        apertium_lexicon_fn = f"noise-annotator/data/tUMPC/lexicon/{language}_apertium.tsv"
        stats_dicts.append(main(train_fn, test_fn, apertium_lexicon_fn, language))


    print("\t".join(stats_dicts[0].keys()))
    for stats_dict in stats_dicts:
        print("\t".join(stats_dict.values()))