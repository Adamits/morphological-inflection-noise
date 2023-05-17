"""Make inflection data to do masked char-level language modeling. We use either:

1. The entire lexicon from the corpus.
2. The entire lexicon of words tUMPC used in a pair.
3. The entire lexicon of training samples that we train on.

We simply have each word as both the input and output of an inflection pair.
During training, masking is applied dynamically at runtime so that each epoch has different parts of a word masked [c.f. RoBERTa]"""

import click
import os


def read_pairs(filename):
    """We assume lines of tab-delimited src, tgt, src_feats, tgt_feats, features"""
    words = set()
    with open(filename, "r") as f:
        for line in f:
            src, tgt, _, _, _ = line.rstrip().split("\t")
            words.add(src)
            words.add(tgt)

    return words


def read_corpus(filename):
    """We assume lines of space delimited tokens"""
    words = set()
    with open(filename, "r") as f:
        for line in f:
            for word in line.rstrip().split():
                words.add(word)

    return words


@click.command()
@click.option("--filename")
@click.option("--out-filename")
@click.option("--format")
def main(filename: str, out_filename: str, format: str):
    """format should be corpus or pairs"""
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    if format == "pairs":
        print(f"Reading pairs from {filename}")
        lexicon = read_pairs(filename)
    elif format == "corpus":
        print(f"Reading words from {filename}")
        lexicon = read_corpus(filename)
    else:
        raise Exception("")

    print(f"Found {len(lexicon)} words")

    print(f"Writing autoencoding pretraining pairs to {out_filename}")
    with open(out_filename, "w") as out:
        for word in lexicon:
            print(f"{word}\t{word}", file=out)


if __name__ == "__main__":
    main()