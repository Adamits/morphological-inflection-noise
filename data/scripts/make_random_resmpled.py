import click
import os

"""replace correct isntances with noisy data from random noise partitions"""


def read(path):
    samples = []
    with open(path, "r") as file:
        for line in file:
            samples.append(line.rstrip())

    return samples


def read_noise(path_prefix, k):
    """Read partitions 1 to k"""
    samples = []
    for i in range(0, k):
        samples.extend(read(f"{path_prefix}{i}.tsv"))

    return samples


@click.command()
@click.option("--corrects-path", required=True)
@click.option("--noise-partitions-path", required=True)
@click.option("--out-dir", required=True)
@click.option("--language", required=True)
@click.option("--num-partitions", required=True, type=int)
def main(corrects_path, noise_partitions_path, out_dir, language, num_partitions):
    corrects = read(corrects_path)
    for k in range(1, num_partitions+1):
        os.makedirs(os.path.join(out_dir, f"{k}"), exist_ok=True)
        with open(os.path.join(out_dir, f"{k}", f"{language}_sampled.tsv"), "w") as out:
            noise = read_noise(os.path.join(noise_partitions_path, f"{language}_noise_"), k)
            if len(noise) > len(corrects):
                data = noise[:len(corrects)]
            else:
                data = corrects[:len(corrects) - len(noise)]
                data += noise

            for sample in data:
                print(sample, file=out)


if __name__ == "__main__":
    main()