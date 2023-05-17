import click
import os
from typing import Dict, List
import csv


def read_csv(fn: str) -> List:
    rows = []
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        # Skip header row
        next(reader, None)
        for row in reader:
            if any(row):
                rows.append(row)

    return rows


def read_tumpc_map(fn: str) -> Dict:
    m = {}
    with open(fn, "r") as f:
        for line in f:
            tumpc_slot, uni_tag = line.rstrip().split("\t")
            m[tumpc_slot] = uni_tag

    return m


def add_msds_to_row(row: List, tumpc_map: Dict) -> List:
    src_word, tgt_word, src_slot, tgt_slot = row
    try:
        src_msd, tgt_msd = tumpc_map[src_slot], tumpc_map[tgt_slot]
    except KeyError:
        slots = []
        if src_slot not in tumpc_map.keys():
            src_msd = "UNK"
            slots.append(src_slot)
        if tgt_slot not in tumpc_map :
            tgt_msd = "UNK"
            slots.append(tgt_slot)

        msg = f"Slots: {'and '.join(slots)} are not mapped. Skipping example\n"
        msg += f"Setting MSDs for '{src_word, tgt_word, src_slot, tgt_slot}' to UNK"
        print(msg)

    return [src_word, tgt_word, src_msd, tgt_msd]


@click.command()
@click.option("--inflections-fn", required=True)
@click.option("--tumpc-map-fn", required=True)
@click.option("--out-fn", required=True)
def main(inflections_fn, tumpc_map_fn, out_fn):
    rows = read_csv(inflections_fn)
    tumpc_map = read_tumpc_map(tumpc_map_fn)
    rows = [add_msds_to_row(row, tumpc_map) for row in rows]
    # Ignore singletons
    rows = [r for r in rows if r[2] != r[3]]

    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    with open(out_fn, "w") as out:
        for row in rows:
            print("\t".join(row), file=out)


if __name__ == "__main__":
    main()