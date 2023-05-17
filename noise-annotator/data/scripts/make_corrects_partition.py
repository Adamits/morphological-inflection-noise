import click

"""Simply filter out all samples not annotated as correct.

Then write the filtered files to a new folder"""


@click.command()
@click.argument("filename", required=True)
@click.argument("output_filename", required=True)
def main(filename, output_filename):
    with open(output_filename, "w") as out:
        with open(filename, "r") as f:
            for line in f:
                if line.rstrip():
                    fields = line.rstrip().split("\t")
                    if fields[-1] == "C":
                        out.write("\t".join(fields))
                        out.write("\n")


if __name__ == "__main__":
    main()