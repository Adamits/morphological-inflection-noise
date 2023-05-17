import os
import click
from typing import Dict, List, Optional, Set, Tuple
import random
import math

import csv

"""Convert the annotated csv into the set of `valid` training samples.
    - Throw out pairs with UNK annotations or ERR annotations
    - Throw out identity inflections
    - Throw out `invalid msds`: those with atomic tags that DO NOT exist in UniMorph for that language
      with the exception of POS tags that occur in POS_TAG_ERRORs
    
    We additionally print statistics for our dataset."""

# If an annotation is exactly any of these, we consider it correct.
# TODO: If we want to use src MSD info, then
# these cannot be considered correct anymore
MAP_TO_CORRECT = set(["SRC_SLOT_ERROR", "MAPPED_POS_PAIR_ERROR"])


def read_annotated(filename: str) -> Dict:
    """Read the auto-annotated pairs and get a dict of all pairs hashed by
    their tuple of annotations

    Args:
        filename (str): _description_

    Returns:
        Dict: _description_
    """
    ret = {}
    count = 0
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        # Header
        next(reader)
        for row in reader:
            # IGNORE IDENTITY INFLECTION.
            # Note this should not effect the 2k samples, since they are 
            # sampled from data that also ignored identity.
            # There is a chance though since some needed to be resampled
            if row[3] == row[5]:
                continue
            annotation = row.pop(-1)
            # FIXME: For now we turn any SRC_SLOT_ERROR-only annotation into "CORRRECT"
            # This is because we ignore src MSDs. If we want to use them, these
            # CANNOT BE CONSIDERED CORRECT.
            annotation = "C" if annotation in MAP_TO_CORRECT else annotation
            # Ignore the apertium msd fields
            row = row [:-2]
            ret.setdefault(annotation, []).append(row)
            count += 1

    print(f"Loaded {count} annotations")
    return ret


def read_valid_tags(filename: str) -> Set[str]:
    """Load the valid MSDs into a set

    Args:
        filename (str)

    Returns:
        Set[str]
    """
    valid = set()
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                valid.add(line.strip())

    return valid


def summarize_annotation_distribution(sampled: List, valid_tags: Set):
    """print a summary of the distribution

    Args:
        sampled (Dict): _description_
        valid_tags_fn (str): _description_
    """
    invalids = set()

    print("Summarizing sampled data...")
    print(f"{len(sampled)} total samples")
    totals_dict = {}
    fine_grained_totals_dict = {}
    invalids_count = 0
    pos_dist = {}
    for sample in sampled:
        pos_dist.setdefault(sample[3].split(";")[0], 0)
        pos_dist[sample[3].split(";")[0]] += 1
        pos_dist.setdefault(sample[5].split(";")[0], 0)
        pos_dist[sample[5].split(";")[0]] += 1
        is_invalid = False
        if sample[3] not in valid_tags:
            invalids.add(sample[3])
            is_invalid = True
        if sample[5] not in valid_tags:
            invalids.add(sample[5])
            is_invalid = True
        if is_invalid:
            invalids_count += 1

        anns = sample[-1]
        fine_grained_totals_dict.setdefault(";".join(anns), 0)
        fine_grained_totals_dict[";".join(anns)] += 1

        modified_ann = ";".join(sorted(set([
            make_simple_tag(a) for a in anns
        ])))
        totals_dict.setdefault(modified_ann, 0)
        totals_dict[modified_ann] += 1

    not_in_uni_percent = round(invalids_count / len(sampled) * 100, 2)
    msg = f"\nFound {invalids_count} ({not_in_uni_percent}%)"
    msg += f" samples with an MSD that does not actually occur in UniMorph"
    print(msg)
    print("Invalid tags:")
    print(invalids)
    print(f"\nOutput dataset full fine-grained annotation distribution:")
    for k, v in sorted(fine_grained_totals_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{k} -- {v} ({round(v/len(sampled) * 100, 2)}%)")

    print(f"\nOutput dataset percents of each coarse grained tag")
    for k, v in sorted(totals_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{k} -- {v} ({round(v/len(sampled) * 100, 2)}%)")

    print("\nDistribution of parts of speech in sampled data:")
    total = sum(pos_dist.values())
    for msd, count in pos_dist.items():
        print(f"{msd} -- {count} ({round(count/total*100, 2)}%)")


def filter_data(annotations: Dict, valid_tags: Set):
    """Filter the annotations dict down to only those samples
    that we were able to provide a valid annotation (i.e. no apertium error, 
    or unknown analysis)

    Args:
        annotated (Dict): _description_
        valid_tags (Set): _description_
    """
    print("Filtering UNK/ERR annotations")
    invalid = []
    filtered_anns = {}
    total = sum([len(d) for d in annotations.values()])
    atomic_valid_tags = set([t for tag in valid_tags for t in tag.split(";")])

    for annotation, data in annotations.items():
        anns = [a for ann in annotation.split(";") for a in ann.split()]
        # Filter out the words whose apertium analysis cannot be trusted
        # Or who have no apertium analysis and were not marked as lexical errors
        if any([a.startswith("ERR") for a in anns]) or anns == ["UNK"]:
            invalid.extend(data)
            continue

        for d in data:
            if not is_valid_sample(d + [annotation], atomic_valid_tags):
                invalid.append(d)
                continue
            # Remove the UNK annotations
            updated_ann = [a for a in anns if not a.startswith("UNK")]
            filtered_anns.setdefault(tuple(updated_ann), []).append(d)

    num_valid = sum([len(v) for v in filtered_anns.values()])
    if not len(invalid) + num_valid == total:
        msg = f"{len(invalid) + num_valid} is not the same as the total, {total}"
        raise Exception(msg)

    msg = f"Removed {len(invalid)}, ({round(len(invalid) / total * 100, 2)}%)"
    msg += " samples (due to Apertium analysis errors, unknown analyses, or invalid MSDs)"
    print(msg)

    num_lex_errs = sum([
        len(d) for a, d in filtered_anns.items() if "SRC_LEXICAL_ERROR" in a or "TGT_LEXICAL_ERROR" in a
    ])
    msg = f"Found {num_lex_errs} lexical errors"
    print(msg)

    return filtered_anns


def remove_annotations(annotations: Dict[Tuple, List], removes: List):

    def _tupelize(ann: str):
        return tuple(sorted(set(ann.split(";"))))

    filtered = {}
    for annotation, data in annotations.items():
        print(annotation)

        simple_ann = tuple(sorted(set([
            make_simple_tag(a) for a in annotation
        ])))
        # Remove any with an annotation that we do not want to sample
        # print(annotation, removes)
        # TODO: For now, we specify full annotation (combinations).
        # We might consider single annotations (e.g. we request SLOT_ERROR be removed
        # and that also removes POS_ERROR;SLOT_ERROR)
        if any([simple_ann == _tupelize(r) for r in removes]): #len(set(simple_ann) & set(removes)) > 0:
            print(f"Ignoring data for {simple_ann}...")
            continue

        filtered.setdefault(annotation, []).extend(data)

    return filtered


def filter_to_include(annotations: Dict[Tuple, List], include_annotations: List):
    def _tupelize(ann: str):
        return tuple(sorted(set(ann.split(";"))))

    filtered = {}
    for annotation, data in annotations.items():
        print(annotation)

        simple_ann = tuple(sorted(set([
            make_simple_tag(a) for a in annotation
        ])))
        # Remove any with an annotation that we do not want to sample
        # print(annotation, removes)
        # TODO: For now, we specify full annotation (combinations).
        # We might consider single annotations (e.g. we request SLOT_ERROR be removed
        # and that also removes POS_ERROR;SLOT_ERROR)
        if any([simple_ann == _tupelize(r) for r in include_annotations]):
            print(f"Including data for {simple_ann}...")
            filtered.setdefault(annotation, []).extend(data)

    return filtered


def make_distribution(anns_dict: Dict):
    """Make a dictionary of the annotation types in `anns_dict`
    hashing their probability mass in the distribution.

    Args:
        anns_dict (Dict): _description_

    Returns:
        _type_: _description_
    """
    num_valid = sum([len(v) for v in anns_dict.values()])
    return {ann: len(v)/num_valid for ann, v in anns_dict.items()}


def load_distribution(spec_fn: str) -> Dict:
    """Load the desired distribution from a spec file

    Args:
        spec_fn (str)

    Raises:
        Exception

    Returns:
        Dict
    """

    ANNS = set([
        "C",
        "MAPPED_POS_PAIR_ERROR",
        "PARADIGM_ERROR",
        "POS_PAIR_ERROR",
        "POS_ERROR",
        "SLOT_ERROR",
        "LEXICAL_ERROR",
        "PARADIGM_ERROR;SLOT_ERROR",
        "POS_ERROR;SLOT_ERROR",
        "POS_PAIR_ERROR;SLOT_ERROR"
    ])

    out = {}
    with open(spec_fn, "r") as file:
        for line in file:
            annotation, pct = line.rstrip().split("\t")

            if annotation not in ANNS:
                msg = f"{annotation} is not a valid annotation. Try one of {', '.join(ANNS)}"
                raise Exception(msg)

            out[annotation] = float(pct)

    if sum(out.values()) != 1.0:
        msg = f"{out.values()} is not a valid distribution! Should sum to 1, not {sum(out.values())}"
        print(f"Warning: {msg}")

    return out


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


def is_valid_sample(sample: List[str], valid_tags: Set) -> bool:
    """Check whether or not a sample has valid tags.

    If any MSD is invalid (i.e. contains atomic tags not in UniMorph) and IS NOT
    marked as a POS_ERROR, then the sample is invalid. Lexical errors are also valid by default.

    Args:
        sample (List[str])
        valid_tags (Set): the atomic tags in the MSDs in UniMorph

    Returns:
        bool
    """
    def _valid_tag(msd, tags):
        msd_tags = set(msd.split(";"))
        if len(msd_tags & tags) == len(msd_tags):
            return True

        return False

    src, tgt, src_slot, src_MSD, tgt_slot, tgt_MSD, annotation = sample

    if "LEXICAL_ERROR" in str(annotation):
        return True

    if not _valid_tag(src_MSD, valid_tags) and "SRC_POS_ERROR" not in annotation:
        return False
    if not _valid_tag(tgt_MSD, valid_tags) and "TGT_POS_ERROR" not in annotation:
        return False

    return True


def sample_data(
    annotations: Dict,
    distribution: Dict,
    k: int
):
    """Sample k pairs from annotations according to the distribution

    Args:
        annotated (Dict)
        distribution (Dict)
        k (int): total amount of data to sample

    Returns:
        List
    """
    total_num_samples = sum([len(a) for a in annotations.values()])

    if distribution is not None:
        print(f"\n Sampling up to {k} samples, out of {total_num_samples} total according to the following distribution:")

        for a, d in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"{a} -- {round(d*100, 2)}%")
        
    sampled = []
    if distribution is not None:
        simple_annotations = {}
        # Now sample each unique annotation combo to produce the same distribution of
        # annotations comprising `k` samples
        for ann, data in annotations.items():
            [d.append(ann) for d in data]
            modified_ann = ";".join(sorted(set([
                make_simple_tag(a) for a in ann
            ])))
            simple_annotations.setdefault(modified_ann, []).extend(data)
    
    # Use the collapsed "simple annotations"
    # Here we still want to maintain the fine-grained annotation,
    # but base the distribution on the simple one
    if distribution is not None:
        print("Using simple annotations for distribution")
        annotations = simple_annotations

    print(f"\n Existing samples distribution:")
    for ann, data in annotations.items():
        print(f"{ann} --- {len(data)} ({round(len(data) / total_num_samples * 100, 2)}%)")

    if distribution is not None:
        for ann, data in annotations.items():
            # Sample size is the rounded up amount according to the distribution.
            # In case this is more than the actual # of that annotation, we just take all of them.
            sample_size = min(math.ceil(distribution.get(ann, 0) * k), len(data))
            # print(len(data), sample_size)
            sampled.extend(random.sample(data, sample_size))
    else:
        # Then we return all data
        sampled = [d + [ann] for ann, data in annotations.items() for d in data]
        random.sample(sampled, min(len(sampled), k))

    return sampled


@click.command()
@click.option("--annotated_fn", type=str, help="Annotated full dataset")
@click.option(
    "--valid_tags_fn", required=True, 
    help="A list of all UniMorph tags for the given language. This is for logging"
)
@click.option("--output_fn", type=str)
@click.option("--k", type=int)
@click.option("--leave_out_annotations", multiple=True)
@click.option("--include-annotations", multiple=True)
@click.option("--distribution_spec", type=str, default=None)
def main(
    annotated_fn,
    valid_tags_fn,
    output_fn,
    k,
    leave_out_annotations,
    include_annotations,
    distribution_spec
):
    if any(leave_out_annotations) and any(include_annotations):
        msg = "Cannot set bor leave_out_annotations and include-annotations options!"
        msg += " include-annotations assumes we include ONLY those annotations."
        raise Exception(msg)
    # 1. Get full annotated dict
    full_anns_dict = read_annotated(annotated_fn)
    valid_tags = read_valid_tags(valid_tags_fn)
    # 2. Do the same for the FULL set
    filtered_anns = filter_data(full_anns_dict, valid_tags)

    if len(leave_out_annotations) > 0:
        print(f"Removing {leave_out_annotations} from the dataset")
        filtered_anns = remove_annotations(filtered_anns, leave_out_annotations)
    elif len(include_annotations) > 0:
        print(f"Including only {include_annotations} in the dataset")
        filtered_anns = filter_to_include(filtered_anns, include_annotations)


    # Load the desired experimental distribution, if any
    dist = None
    if distribution_spec:
        dist = load_distribution(distribution_spec)
    # 6. Sample according to the distribution from the FULL set.
    sampled = sample_data(
        filtered_anns,
        dist,
        k
    )
    summarize_annotation_distribution(sampled, valid_tags)

    random.shuffle(sampled)
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    print(f"Writing {len(sampled)} total samples to {output_fn}")
    with open(output_fn, "w") as out:
        for sample in sampled:
            src, tgt, _, src_msd, _, tgt_msd, anns = sample
            anns = ";".join(anns)
            print(f"{src}\t{tgt}\t{src_msd}\t{tgt_msd}\t{anns}", file=out)



if __name__ == "__main__":
    main()