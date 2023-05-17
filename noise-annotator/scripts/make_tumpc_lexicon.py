import click

"""Read the tUMPC paradigms, marked with old mappings, etc.

Then generate a lexicon tsv of word \t slot_id"""


def read_tumpc(fn):
    ret = {}

    with open(fn, "r") as f:
        # First line is header
        next(f)
        for line in f:
            line = line.rstrip()

            if not line:
                continue

            src, tgt, src_slot, tgt_slot = line.split(",")
            
            ret.setdefault(src, set()).add(src_slot)
            ret.setdefault(tgt, set()).add(tgt_slot)

    return ret


@click.command()
@click.option("--fn", required=True)
@click.option("--outfn", required=True)
def main(fn, outfn):
    print(f"Reading {fn}")
    lexicon = read_tumpc(fn)

    print(f"writing to {outfn}")
    with open(outfn, "w") as o:
        for word, tags in lexicon.items():
            if len(tags) > 1:
                print(f"{word} has {' and '.join(tags)}")
            
            for t in tags:
                o.write("\t".join([word, t]))
                o.write("\n")


if __name__=='__main__':
    main()