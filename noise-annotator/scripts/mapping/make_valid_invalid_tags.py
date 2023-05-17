import click
import os
import subprocess
from typing import List, Set

from tqdm import tqdm

"""Read the language mapping, and apply apertium to every word in the UniMorph DB for a given language

NOTE: We ignore words that include whitespace (MWEs), since these are tricky for parsing apertium outputs, and tUMPC does not include them anyway

We then report accuracy on mapping apertium analyses to UniMorph tags, assuming if there is any intersection in analyses for a given word, then it is correct. This is because apertium can be a superset of UniMorph tags."""


def read_map(fn):
    map = {}
    order = []
    with open(fn, "r") as f:
        for line in f:
            line = line.rstrip()

            if line:
                apt, uni = line.split("\t")
                map[apt] = uni
                if uni not in order:
                    order.append(uni)

    return map, order


def read_uni(fn):
    # Map each word to all its possible analyses
    word_tags = {}
    with open(fn, "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                _, word, msd = line.split("\t")
                # IGNORE MULTI-WORD EXPRESSIONS
                if " " not in word:
                    word_tags.setdefault(word, set()).add(msd)

    return word_tags


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

    # if any(cmps):
    #     print("COMPOUND")
    #     print(cmps)
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


def apply_rus_heuristics(s):
    # Active and Passive only marked on participles
    # Need to split participle tense info to insert act/pass in correct spot
    if s.startswith("V.PTCP;"):
        if ";ACT" in s:
            s = s.split(";")
            s = ";".join([s[0], "ACT", s[1]])
        elif ";PASS" in s:
            s = s.split(";")
            s = ";".join([s[0], "PASS", s[1]])
        else:
            s = s#"V.PTCP;PST"
    else:
        s = s.replace(";ACT", "")
        s = s.replace(";PASS", "")

    # In verbs, gender comes before number unlike Noun/ADJ
    if s.startswith("V;PST"):
        if any([g in s for g in ["MASC;", "FEM;", "NEUT;"]]):
            s = s.split(";")
            s = ";".join(s[:-2] + [s[-1], s[-2]])

    if not (s.startswith("ADJ;") or s.startswith("V;PST")):
        s = s.replace(";MASC", "")
        s = s.replace(";FEM", "")
        s = s.replace(";NEUT", "")

    if not "ACC;MASC;SG" in s or not (";ACC" not in s and ";PL" not in s):
        s = s.replace(";INAN", "")
        s = s.replace(";ANIM", "")

    # If no case, assume ESS
    if s.startswith("N;") and not (
        "NOM" in s or
        "DAT" in s or
        "ACC" in s or
        "GEN" in s or
        "ESS" in s or
        "INS" in s
    ):
        s = s.replace("N;", "N;ESS;")
    elif s.startswith("ADJ;") and not (
        "NOM" in s or
        "DAT" in s or
        "ACC" in s or
        "GEN" in s or
        "ESS" in s or
        "INS" in s or
        "LGSPEC1" in s
    ):
        s = s.replace("ADJ;", "ADJ;ESS;")

    return s


def apply_swe_heuristics(s):
    if s.startswith("V.PTCP;PST"):
        s = "V.PTCP;PST"

    if not s.startswith("ADJ;"):
        if s.startswith("N;"):
            # NEUT and MASC+FEM not marked on UM nouns
            # We also want to default to nominative in this case
            s = s.replace("MASC+FEM;", "NOM;")
            s = s.replace("NEUT;", "NOM;")
            if "DAT;" in s or "ACC;" in s or "GEN;" in s:
                s = s.replace("NOM;", "")
        else:    
            s = s.replace("MASC+FEM;", "")

    s = s.replace(";IMP;2", ";IMP")

    return s


def apply_deu_heuristics(s):
    if s.startswith("V.PTCP;PST"):
        return "V.PTCP;PST"

    # Nominalization from gerunds
    if "GER" in s:
        s = s.replace("V", "N")
        s = s.replace("GER;", "")
        # Gerunds are singular
        s = s + ";SG"

    return s


def apply_fin_heuristics(s):
    if s.startswith("V.PTCP;"):
        if ";ACT" in s:
            s = s.split(";")
            s = ";".join([s[0], "ACT", s[1]])
        elif ";PASS" in s:
            s = s.split(";")
            s = ";".join([s[0], "PASS", s[1]])
        else:
            s = s#"V.PTCP;PST"

    if ";NFIN" in s:
        s = "V;NFIN"

    if s.startswith("ADJ;"):
        s = s.replace("GEN", "GEADJ")

    # Finish COM case defaults to PL.
    # If apt also gets a sg tag, pl default seems better still
    if "PL;SG" in s:
        s = s.replace("PL;SG", "PL")

    return s


def apply_isl_heuristics(s):
    s = s.replace("2;2", "2")

    if s.startswith("V;") and not (
        "NFIN" in s or
        "IND" in s or
        "SBJV" in s or
        "IMP" in s
    ):
        s = s.replace("V;", "V;IND;")

    return s


def apply_heuristics(s, lang):
    D = {
        "rus": apply_rus_heuristics,
        "swe": apply_swe_heuristics,
        "deu": apply_deu_heuristics,
        "fin": apply_fin_heuristics,
        "isl": apply_isl_heuristics
    }

    if not lang in D:
        print(f"No heuristics for {lang}")
        return s

    return D[lang](s)


def apply_map(lang, analysis, tags_map, order):
    tags = [tags_map[t] for t in analysis.split(";") if tags_map.get(t)]
    # Order by when the unimorph tag occured int he file. 
    # This is to endorce the order of tags so we get UniMorph MSDs
    tags = [o for o in order if o in tags]

    # Post-processing removal of duplicates that can come
    # from the one-to-many mappings s.t. one `tag` here actaully comprised 2 tags
    # e.g. tag="V.PTCP;DEF"
    seen = set()
    outs = [t for tag in tags for t in tag.split(";") if not (t in seen or seen.add(t))]
    outs = [";".join(outs)]

    # SOME UNIVERSAL HEURISTICS AND SPLITTING INTO MULTIPLE TAGS
    if "V;" in outs[0] and "V." in outs[0]:
       outs[0] = outs[0].replace("V;", "")
    if "V.CVB;" in outs[0] and "V.PTCP;" in outs[0]:
       outs[0] = outs[0].replace("V.PTCP;", "")
    # Apt sometimes marks adj on participles. Here we just produce both corresponding unimorph tags.
    if "ADJ;" in outs[0] and "V.PTCP" in outs[0]:
       outs = [outs[0].replace("ADJ;", ""), outs[0].replace("V.PTCP;", "")]
    # For apertium SG/PL in e.g. Swedish, we just generate both versions 
    # since sometimes unimorph defaults to one.
    elif "SG;PL" in outs[0]:
        outs = [outs[0].replace(";SG;PL", ";SG"), outs[0].replace(";SG;PL", ";PL")]

    for i, o in enumerate(outs):
        # If we replace with a past participle, we remove e.g. person/number/gender info
        if o.startswith("ADJ;"):
            o = o.replace(";PST", "")

        outs[i] = apply_heuristics(o, lang)

    return [o.rstrip(";") for o in outs]


def equal_msd(apt_msds: Set[str], uni_msds: Set[str]):
    # Check if they intersect, i.e. agree on any MSDs
    return len(apt_msds & uni_msds) > 0


@click.command()
@click.option("--map_fn", required=True)
@click.option("--uni_fn", required=True)
@click.option("--apertium_fn", required=True)
@click.option("--outpath", required=True)
def main(map_fn, uni_fn, apertium_fn, outpath):
    lang = os.path.basename(uni_fn).split(".")[0]
    # 1. read map
    map, order = read_map(map_fn)
    # print(apply_map(lang, "vblex;ger;acc", map, order))
    # raise
    # 2. Read UniMorph
    uni_word_msds = read_uni(uni_fn)
    # 3. For each, map to UniMorph and check if tags match
    corr, tot = 0, 0
    print("Analyzing words...")
    all_analyses = run_batch_analyzer(apertium_fn, [w for w in uni_word_msds.keys()])
    
    valid_tags = set().union(*uni_word_msds.values())
    invalid_tags = set()
    tag_cnts = {}
    invalid_msd_tag_count = 0
    invalid_msds = set()
    inv_msd_map = {}
    print("Comparing with msds...")
    for (word, uni_msds), analyses in zip(uni_word_msds.items(), all_analyses):
        # Remove lemmas
        analyses = [";".join(a.split(";")[1:]) for a in analyses]
        # analyses = run_analyzer(apertium_fn, word)
        apt_msds = set()
        for a in analyses:
            # [apt_msds.add(m) for m in apply_map(lang, a, map, order)]
            for m in apply_map(lang, a, map, order):
                inv_msd_map.setdefault(m, set()).add(a)
                apt_msds.add(m)

        apt_msds = set([a for a in apt_msds if a])
        for m in apt_msds:
            if m not in valid_tags:
                invalid_msds.add(m)
                invalid_msd_tag_count += 1
        # print(word, apt_msds, msd)
        # Get acc in cases that apertium actually predicted an analysis
        if any(apt_msds):
            tag_cnts.setdefault(",".join(analyses), [0, 0])
            tag_cnts[",".join(analyses)][1] += 1
            #if msd in apt_msds:
            if equal_msd(apt_msds, uni_msds):
                corr += 1
            else:
                # print(word, apt_msds, uni_msds)
                invalid_tags.add(",".join(analyses))
                tag_cnts[",".join(analyses)][0] += 1

            tot += 1

    print(f"Found {invalid_msd_tag_count} words with MSDs not in UniMorph:")
    print(invalid_msds)
    for msd in invalid_msds:
        print(msd, inv_msd_map[msd])
    print(f"{tot - corr} disagreements out of {tot}")
    print(round(100 * corr/tot, 2))

    print(f"Writing valid/invalid tags to {outpath}")
    with open(outpath + f"/{lang}_valid_tags.txt", "w") as o:
        o.write("\n".join(valid_tags))

    # [OLD] Prune invalid tags to those that error at least 10% of the time
    pruned_invalid = set()
    for t in invalid_tags:
        # print(t, round(tag_cnts[t][0] / tag_cnts[t][1], 3), tag_cnts[t][1])
        # NEVERMIND LETS INCLUDE EVERYING
        pruned_invalid.add(t)
        # if tag_cnts[t][0] / tag_cnts[t][1] >= 0.1:
        #     pruned_invalid.add(t)

    with open(outpath + f"/{lang}_invalid_tags.txt", "w") as o:
        o.write("\n".join(pruned_invalid))


if __name__=='__main__':
    main()