import click
import os
import subprocess
from typing import List, Set

from tqdm import tqdm

"""Read the language mapping, and apply it to the analyses in the apertium lexicon in order to get the corresponding UniMorph tags

Note that we filter out words from the apertium lexicon with an analysis that ever disagreed with UniMorph
Or words whose mapped tag does not exist in UniMorph"""


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


def read_lexicon(fn, invalid_tags):
    lex = {}
    with open(fn, "r") as f:
        for line in f:
            word, *tags = line.rstrip().split("\t")
            lex.setdefault(word, set())
            # If apertium has no tag for it due to no analysis,
            # Then just set an UNK tag
            if not tags or tags[0].startswith("*"):
                tags = "UNK"
            else:
                # Remove lemmas
                new_tags = []
                for a in tags[0].split(","):
                    new_tags.append(";".join(a.split(";")[1:]))
                
                # for t in new_tags:
                #     if t.split(";")[0] not in ["adj", "n", "vblex"]:
                #         print(t)
                tags = ",".join(new_tags)
                if tags in invalid_tags:
                    tags = "ERR"

            [lex[word].add(t) for t in tags.split(",")]

    return lex


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
    if analysis == "UNK" or analysis == "ERR":
        return [analysis]
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


def read_tags(fn):
    tags = set()
    with open(fn, "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                for tag in line.split(","):
                    tags.add(line)

    return tags


@click.command()
@click.option("--map_fn", required=True)
@click.option("--apertium_fn", required=True)
@click.option(
    "--invalid_tags_fn", required=True, 
    help="A list of apertium analyses that disagreed with UniMorph in tests"
)
@click.option(
    "--valid_tags_fn", required=True, 
    help="A list of all UniMorph tags for the given language"
)
@click.option("--out_fn", required=True)
def main(map_fn, apertium_fn, invalid_tags_fn, out_fn, valid_tags_fn):
    lang = os.path.basename(apertium_fn).split("_")[0]
    print(f"Applying map for {lang}...")
    tmap, order = read_map(map_fn)
    invalid_tags = read_tags(invalid_tags_fn)
    valid_tags = read_tags(valid_tags_fn).union(set(["UNK", "ERR"]))
    # Invalid tags will be mapped to `ERR`
    lex = read_lexicon(apertium_fn, invalid_tags)

    print(f"Writing to {out_fn}")
    with open(out_fn, "w") as o:
        for word, tags in lex.items():
            mapped_tags = set()
            for t in tags:
                mapped_tags = mapped_tags.union(set(apply_map(lang, t, tmap, order)))
            # We remove mappings that lead to a tag that never actually occurs in UniMorph.
            # This is because our dataset needs to share its tag inventory with UniMorph
            # mapped_tags = [mt for mt in mapped_tags if mt in valid_tags]
            # # If this leaves no mapping, then we set it to ERR
            # if not mapped_tags:
            #     mapped_tags = ["ERR"]
            # print(valid_tags)
            for mt in mapped_tags:
                # Sometimes heuristics might turn a weird looking tag into the empty string.
                if not mt:
                    mt = "UNK"

                o.write(f"{word}\t{mt}")
                o.write("\n")

if __name__=='__main__':
    main()