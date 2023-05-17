from typing import List
import random
import click

"""Take the resampled SIGMORPHON samples, and change them from lemma -> surface format
to surface -> surface format to keep as similar as possible to tUMPC data.

We make sure to keep the target the same, and just resample a different surface form instead
of using the lemma."""


def read_samples(fn: str):
    samples = []

    with open(fn, "r") as f:
        for line in f:
            if line.rstrip():
                samples.append(line.rstrip().split("\t"))

        return samples


def read_paradigms(fn: str):
    paradigms = {}

    with open(fn, "r") as f:
        for line in f:
            if line.rstrip():
                lemma, tgt, msd = line.rstrip().split("\t")
                paradigms.setdefault(lemma, []).append((tgt, msd))

    return paradigms


def resample_surface(paradigm: List[str], lemma: str, src_msd: str, tgt_msd: str):
    candidates = [p for p in paradigm if p[1] != tgt_msd]
    # DO NOT SAMPLE FORMS WITH WHITESPACE (e.g. multiple word inflections)
    candidates = [p for p in paradigm if " " not in p[0] and " " not in p[1]]

    if len(candidates) < 1:
        print(f"No unique canidates for {paradigm}. Keeping lemma: {lemma, src_msd}")
        print("(TODO: Should we skip it?)")
        return lemma, src_msd

    random.shuffle(candidates)
    return candidates[0]


@click.command()
@click.option("--filename", required=True)
@click.option("--unimorph_filename", required=True)
@click.option("--output_filename", required=True)
def main(filename, unimorph_filename, output_filename):
    replaced_count = 0
    samples = read_samples(filename)
    paradigms = read_paradigms(unimorph_filename)

    with open(output_filename, "w") as out:
        for src, tgt, src_msd, tgt_msd, ann in samples:
            if ann == "C" and src_msd.endswith(";NFIN"):
                src, src_msd = resample_surface(paradigms[src], src, src_msd, tgt_msd)
                replaced_count += 1

            out.write("\t".join([src, tgt, src_msd, tgt_msd, ann]))
            out.write("\n")


if __name__ =="__main__":
    main()