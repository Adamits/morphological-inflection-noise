import subprocess
import click
import os
from typing import List

from tqdm import tqdm

"""Read the tUMPC lexicons of word \t slot_id

Then generate a lexicon tsv of word \t Apertium tags"""


# APT_CASE_TAGS = set([
#     "<nom>",
#     "<dat>",
#     "<acc>",
#     "<gen>",
#     "<par>",
#     "<ess>",
#     "<ine>",
#     "<com>",
#     "<ill>",
#     "<all>",
#     "<abe>",
#     "<ade>",
#     "<ins>",
#     "<tra>",
#     "<abl>",
#     "<ela>"
# ])


def format_compound(a) -> str:
    #^tulija/tulija<n><sg><nom>/tulla<vblex>+ja<n><sg><nom>$
    def _get_lem(c: str):
        return c.split("<")[0]

    def _get_tags(c: str) -> List:
        return [t.rstrip(">") for t in c.split("<")[1:]]

    cmps = a.split("+")
    # Lemma
    a = ["+".join([_get_lem(c) for c in cmps])]
    # Add tags, taking those of the LAST component in the compound/derived form
    a.extend(_get_tags(cmps[-1]))

    return ";".join(a)

def format_analysis(analyses: List):
    """Format the apertium analysis into comma delimited string of ;-delimited tags"""
    frmtd_analysis = []
    # To track derived/compound versus lexicalized analysis
    cmps = []
    for a in analyses:
        if "+" in a:
            cmps.append(a)
        else:
            frmtd_analysis.append(
                ";".join(a.rstrip("$").replace(">", "").split("<"))
            )

    # If there are lexicalized analyses, just use those and ignore compound/derived analysis
    if any(frmtd_analysis):
        return frmtd_analysis
    # Otherwise, we can return the compound analyses
    else:
        for a in cmps:
            frmtd_analysis.append(format_compound(a))
    
    return frmtd_analysis


def parse_analysis(aprt: str):
    aprt = aprt.split("$")[0]
    inp, *analyses = aprt.rstrip().split("/")

    frmtd_analysis = format_analysis(analyses)

    return frmtd_analysis


def run_batch_analyzer(analyzer_path: str, words: List) -> str:
    """Run the apertium analyzer, return the string of analysis"""

    all = []
    process = "hfst-proc" if analyzer_path.endswith(".hfst") else "lt-proc"
    
    words_str = "\n".join(words)
    with open("./tmp_lt_words.txt", "w") as o:
        o.write(words_str + "\n")

    cmd = f"{process} -a {analyzer_path} ./tmp_lt_words.txt"

    subp = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    outputs, error = subp.communicate()
    if error is not None:
        raise Exception(error)
    
    for o in outputs.decode("utf-8").split("\n")[:-1]:
        all.append(parse_analysis((o)))

    return all


def read_lexicon(fn):
    with open(fn, "r") as f:
        for line in f:
            w, t = line.rstrip().split("\t")
            yield w


@click.command()
@click.option("--lexicon_fn", required=True)
@click.option("--analyzer_fn", required=True)
@click.option("--outfn", required=True)
def main(lexicon_fn, analyzer_fn, outfn):
    words = [w for w in read_lexicon(lexicon_fn)]
    with open(outfn, "w") as out:
        for i, analysis in enumerate(run_batch_analyzer(analyzer_fn, words)):
            out.write(f"{words[i]}\t{','.join(analysis)}\n")


if __name__=='__main__':
    main()