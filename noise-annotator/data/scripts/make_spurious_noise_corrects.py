import click

"""Reannotates any sample with ONLY a SRC_SLOT_ERROR or MAPPED_POS_PAIR_ERROR 
as correct.

This is because we test on inflection, ignoring the src tag, so these samples cannot be marked as an error."""

REANNOTATES = set(["SRC_SLOT_ERROR", "MAPPED_POS_PAIR_ERROR"])


@click.command()
@click.argument("filename", required=True)
@click.argument("output_filename", required=True)
def main(filename, output_filename):
    with open(output_filename, "w") as out:
        with open(filename, "r") as f:
            for line in f:
                if line.rstrip():
                    fields = line.rstrip().split("\t")
                    if fields[-1] in REANNOTATES:
                        fields[-1] = "C"

                    out.write("\t".join(fields))
                    out.write("\n")


if __name__ == "__main__":
    main()