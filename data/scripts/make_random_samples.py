import click
import random
import math
import os

"""Sample k percent of full training data, and write to a new file."""


def read(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            src, tgt, src_msd, tgt_msd, anns = line.rstrip().split("\t")
            samples.append([src, tgt, src_msd, tgt_msd, anns])

    return samples


def write(data, path):
    print(f"Writing {len(data)} samples to {path}")
    with open(path, "w") as o:
        for sample in data:
            print("\t".join(sample), file=o)


@click.command()
@click.option("--data-path", required=True)
@click.option("--k", type=float, required=True)
@click.option("--outfile", required=True)
def main(data_path, k, outfile):
    """Read data and write out a randomly sampled set of
    k% of samples to outfile

    Args:
        data_path (str): The full training set.
        k (float): The percent of training data to sample.
        outfile (str): The file to write k% of training data to.
    """
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    data = read(data_path)
    print(f"Sampling {k*100}% of the {len(data)} training samples...")
    n = math.ceil(k * len(data))
    sample = random.sample(data, n)
    write(sample, outfile)


if __name__ == "__main__":
    main()