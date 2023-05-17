import os
import click


def read_wordlist(fn):
    words = set()
    with open(fn, "r") as f:
        for line in f:
            words.add(line.rstrip().lower())

    return words


@click.command()
@click.option("--wordlist_fn", type=str, required=True)
@click.option("--wikipath", type=str, required=True)
@click.option("--outfn", type=str, required=True)
def main(wordlist_fn, wikipath, outfn):
    realwords = set()
    lex_errors = read_wordlist(wordlist_fn)
    for dir in os.listdir(wikipath):
        for fn in os.listdir(os.path.join(wikipath, dir)):
            wiki_words = set()
            path = os.path.join(wikipath, dir, fn)

            with open(path, "r") as f:
                 for line in f:
                     words = line.rstrip().lower().split(" ")
                     wiki_words |= set(words)

            new_realwords = lex_errors & wiki_words
            lex_errors = lex_errors - new_realwords
            realwords |= new_realwords

    with open(outfn + ".lex_errors", "w") as o:
        for e in lex_errors:
            o.write(e)
            o.write("\n")

    with open(outfn + ".realwords", "w") as o:
        for r in realwords:
            o.write(r)
            o.write("\n")


if __name__=='__main__':
    main()