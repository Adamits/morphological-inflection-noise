"""Sample k training samples from the full `baseline` set of annotated tUMPC data.
Keep the same distribution for each noise annotation.

This enables experiments where we fix the data size to be the same as the all-corrects training
partition, to measure the impact of replacing noise.
"""

from typing import Dict, List, Tuple
import click
import random
import math


def read_baseline_data(filename: str) -> List[List[str]]:
    samples = []
    with open(filename, "r") as f:
        for line in f:
            src, tgt, src_msd, tgt_msd, anns = line.rstrip().split("\t")
            samples.append([src, tgt, src_msd, tgt_msd, anns])

    return samples


def compute_distribution(data: List[List[str]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    dist = {}
    samples = {}
    tot = 0
    for src, tgt, src_msd, tgt_msd, anns in data:
        samples.setdefault(anns, []).append([src, tgt, src_msd, tgt_msd, anns])
        dist.setdefault(anns, 0)
        dist[anns] += 1
        tot += 1

    return {k: v/tot for k, v in dist.items()}, samples


def sample_data(data: Dict[str, List[str]], dist: Dict[str, float], k: int) -> List[List[str]]:
    out = []
    print(f"sampling {k} according to the following distribution:")
    for ann, pct in dist.items():
        print(f"{ann}: {round(pct * 100, 2)}%")
        out.extend(random.sample(data[ann], math.ceil(k * pct)))

    # random.shuffle(out)
    # Resample from this distribution to get the exact size
    return random.sample(out, k)


@click.command()
@click.option("--input-filepath")
@click.option("--output-filepath")
@click.option("--k", type=int)
def main(input_filepath, output_filepath, k):
    data = read_baseline_data(input_filepath)
    dist, ann2data = compute_distribution(data)
    output_data = sample_data(ann2data, dist, k)

    print(f"Found {len(output_data)} samples")
    with open(output_filepath, "w") as out:
        for sample in output_data:
            print("\t".join(sample), file=out)


if __name__ == "__main__":
    main()