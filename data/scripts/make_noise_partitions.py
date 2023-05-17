"""We make 10 evenly sized partitions of noisy samples, without resampling.

These can be used to form datasets that show the effect of adding noise on accuracy."""

import math
import os
import random


def read_noise(data_path):
    samples = []
    with open(data_path) as f:
        for line in f:
            src, tgt, src_msd, tgt_msd, ann = line.rstrip().split("\t")

            if ann != "C":
                samples.append((src, tgt, src_msd, tgt_msd, ann))

    return samples


def main(data_path, outdir, lang):
    os.makedirs(outdir, exist_ok=True)
    noisy_samples = read_noise(data_path)
    random.shuffle(noisy_samples)
    size = math.ceil(len(noisy_samples)/10)

    partitions = []
    partition = []
    for i, s in enumerate(noisy_samples):
        partition.append(s)
        if i > 0 and (i+1) % size == 0:
            partitions.append(partition)
            partition = []
    
    partitions.append(partition)

    for i, p in enumerate(partitions):
        with open(os.path.join(outdir, f"{lang}_noise_{i}.tsv"), "w") as o:
            for sample in p:
                print("\t".join(sample), file=o)


if __name__ == "__main__":
    DATAPATH = "/Users/adamwiemerslage/nlp-projects/morphology/noisy-inflection/noise-annotator/data/"
    for lang in ["deu", "isl", "swe", "rus"]:
        # main(
        #     DATAPATH + f"sampled/baseline/msd_sampled/{lang}_sampled_sigmorphon_resampling_reinflection.tsv",
        #     DATAPATH + f"sampled/baseline/msd_sampled/random_noise_partitions",
        #     lang
        # )
        main(
            DATAPATH + f"sampled/baseline/{lang}_sampled.tsv",
            DATAPATH + f"sampled/baseline/random_tumpc_corrects_partitions",
            lang
        )