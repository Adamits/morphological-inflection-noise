"""Sample from the full partition that is already formatted (as opposed to sampling from the CSV)

This allows us to easily make add-one-in/leave-one-out based on any set of corrects (e.g. sigmorphon-reinflection-resampled)."""

import os
import click
from typing import Dict, List, Tuple


def make_simple_tag(ann: str) -> str:
    """strip fine-grained tags to braod categories ignoreing src vs tgt differences

    Args:
        ann (str)

    Returns:
        str
    """
    if ann.startswith("SRC_"):
        return ann.replace("SRC_", "")
    elif ann.startswith("TGT_"):
        return ann.replace("TGT_", "")
    else:
        return ann


def remove_annotations(samples: List[Tuple], removes: List):

    def _tupelize(ann: str):
        return tuple(sorted(set(ann.split(";"))))

    filtered = []
    for sample in samples:
        annotation = sample[-1]

        simple_ann = tuple(sorted(set([
            make_simple_tag(a) for a in annotation
        ])))
        # Remove any with an annotation that we do not want to sample
        # print(annotation, removes)
        # TODO: For now, we specify full annotation (combinations).
        # We might consider single annotations (e.g. we request SLOT_ERROR be removed
        # and that also removes POS_ERROR;SLOT_ERROR)
        if any([simple_ann == _tupelize(r) for r in removes]): #len(set(simple_ann) & set(removes)) > 0:
            continue

        filtered.append(sample)

    return filtered


def filter_to_include(samples: List[Tuple], include_annotations: List):
    def _tupelize(ann: str):
        return tuple(sorted(set(ann.split(";"))))

    filtered = []
    for sample in samples:
        annotation = sample[-1]
        simple_ann = tuple(sorted(set([
            make_simple_tag(a) for a in annotation
        ])))
        # Remove any with an annotation that we do not want to sample
        # print(annotation, removes)
        # TODO: For now, we specify full annotation (combinations).
        # We might consider single annotations (e.g. we request SLOT_ERROR be removed
        # and that also removes POS_ERROR;SLOT_ERROR)
        if any([simple_ann == _tupelize(r) for r in include_annotations]):
            filtered.append(sample)

    return filtered


def read_data(filename: str) -> List[str]:
    samples = []
    with open(filename, "r") as f:
        for line in f:
            sample = line.rstrip().split("\t")
            sample[-1] = sample[-1].split(";")
            samples.append(sample)

    return samples


def summarize(data):
    annotation_counts = {}
    total = 0
    for sample in data:
        a = ";".join(sample[-1])
        annotation_counts.setdefault(a, 0)
        annotation_counts[a] += 1
        total += 1

    dist = {a: c/total for a, c in annotation_counts.items() }

    for a, pdf in dist.items():
        print(f"{a} -- {round(pdf * 100, 2)}%")


@click.command()
@click.option("--data-path")
@click.option("--output-path")
@click.option("--leave_out_annotations", multiple=True)
@click.option("--include-annotations", multiple=True)
def main(data_path, output_path, leave_out_annotations, include_annotations):
    data = read_data(data_path)
    if len(leave_out_annotations) > 0:
        print(f"Removing {leave_out_annotations} from the dataset")
        data = remove_annotations(data, leave_out_annotations)
    elif len(include_annotations) > 0:
        print(f"Including only {include_annotations} in the dataset")
        data = filter_to_include(data, include_annotations)

    summarize(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing {len(data)} total samples to {output_path}")
    with open(output_path, "w") as out:
        for sample in data:
            src, tgt, src_msd, tgt_msd, anns = sample
            anns = ";".join(anns)
            print(f"{src}\t{tgt}\t{src_msd}\t{tgt_msd}\t{anns}", file=out)


if __name__ == "__main__":
    main()