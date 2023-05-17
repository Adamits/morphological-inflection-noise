# Reads the lines of a sampled file, and then searches for the line in the updated 
# (read: preserving case and reducing columns) version of the line.
# Finally, writes a new CSV comprising the updated lines to tUMPC/full

import random
from collections import Counter

def read_sampled(fn):
    samples = []
    with open(fn, "r") as f:
        next(f)
        for line in f:
            src, tgt, src_slot, src_pred_msd, tgt_slot, tgt_pred_msd, *ann = line.rstrip().split(",")
            samples.append([src, tgt, src_slot, src_pred_msd, tgt_slot, tgt_pred_msd, *ann])

    return samples


def read_full(fn):
    samples = []
    with open(fn, "r") as f:
        next(f)
        for line in f:
            src, tgt, src_slot, tgt_slot = line.rstrip().split(",")
            samples.append([src, tgt, src_slot, tgt_slot])

    return samples


def find(sample, full):
    method = ""
    for i, full_sample in enumerate(full):
        if sample[0].lower() == full_sample[0].lower() \
            and sample[1].lower() == full_sample[1].lower():
            # print("Found match!")
            # print(sample[:4], full_sample)
            method = "exact"
            return full.pop(i), method

    for i, full_sample in enumerate(full):
        if sample[1].lower() == full_sample[1].lower():
            # print("Found match!")
            # print(sample[:4], full_sample)
            method = "tgt"
            return full.pop(i), method

    for i, full_sample in enumerate(full):
        if sample[0].lower() == full_sample[0].lower():
            # print("Found match!")
            # print(sample[:4], full_sample)
            method = "src"
            return full.pop(i), method

    full_sample = random.sample(full, 1)
    method = "random"
    return full.pop(full.index(full_sample[0])), method

def main(sampled_fn, full_fn, output_fn):
    # 1. read sampled 2k file
    samples = read_sampled(sampled_fn)
    # 2. search for that line in full csv
    fulls = read_full(full_fn)
    # 3. Write the updated version of the line
    match_methods = []
    with open(output_fn, "w") as out:
        for i, sample in enumerate(samples):
            updated_sample, match_method = find(sample, fulls)
            match_methods.append(match_method)
            print(",".join(updated_sample), file=out)

    
    counts = Counter(match_methods)
    print(f"Counts of how each match was found: {counts}")

if __name__ == "__main__":
    sampled_fn="data/tUMPC/sampled/sv_2k.csv"
    full_fn="data/tUMPC/full/swe.csv"
    output_fn="data/tUMPC/full/swe_2k_TEST.csv"
    main(sampled_fn, full_fn, output_fn)