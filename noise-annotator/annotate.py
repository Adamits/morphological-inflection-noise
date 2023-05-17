from tqdm import tqdm
import phunspell

import click
import csv
import subprocess
from typing import Dict, Generator, List, Set

"""CHANGELOG

UPDATE 9/13:
    - seperate lexical error/unk detection
    - use wiki filtered AND dictionary library to do full lexical error detection here
    - Add slot detection to this script
"""


SRC_UNK = "SRC_UNK"
TGT_UNK = "TGT_UNK"
SRC_LEXICAL_ERROR = "SRC_LEXICAL_ERROR"
TGT_LEXICAL_ERROR = "TGT_LEXICAL_ERROR"
POS_PAIR_ERROR = "POS_PAIR_ERROR"
SRC_POS_ERROR = "SRC_POS_ERROR"
TGT_POS_ERROR = "TGT_POS_ERROR"
PARADIGM_ERROR = "PARADIGM_ERROR"
CORRECT = "C"

# SLOT ERRORS
SRC_SLOT_ERROR = "SRC_SLOT_ERROR"
TGT_SLOT_ERROR = "TGT_SLOT_ERROR"
# The words represent a valid inflection, and the slots are possible 
# for both words, but do not match in POS
MAPPED_POS_PAIR_ERROR = "MAPPED_POS_PAIR_ERROR"
# These flag that something went wrong...
MANUALLY_CHECK_SRC_SLOT_ERROR = "MANUALLY_CHECK_SRC_SLOT_ERROR"
MANUALLY_CHECK_TGT_SLOT_ERROR = "MANUALLY_CHECK_TGT_SLOT_ERROR"

# These are for matching the apertium lexicon.
# They flag that a word cannot be reliably mapped to UniMorph
INVALID_APT_TAGS = {"UNK", "ERR"}

# We only consider these POS to be valid inflections
# Otherwise its a POS_ERROR
ISL_VALID_POS = set(["verb", "adj", "n"]) # TODO: 'prn'?
DEU_VALID_POS = set(["verb", "adj", "n", "prn"])
RUS_VALID_POS = set(["prn", "num", "verb", "adj", "n"])
SWE_VALID_POS = set(["verb", "adj", "n"])
FIN_VALID_POS = set(["prn", "num", "verb", "adj", "adv", "n"])

valid_pos_fac = {
    "isl": ISL_VALID_POS,
    "deu": DEU_VALID_POS,
    "rus": RUS_VALID_POS,
    "swe": SWE_VALID_POS,
    "fin": FIN_VALID_POS,
}


APERT_POS_TO_UNI = {
    "verb": "V",
    "vblex": "V",
    "vaux": "V",
    "vbmod": "V",
    "vbser": "V",
    "vbhaver": "V",
    "n": "N",
    "np": "PROPN",
    "adj": "ADJ",
    "adv": "V.CVB",
    "prn": "PRN",
    "num": "NUM",
    "det": "DET",
    "pr": "PREP",
    "pprs": "V.PTCP",
    "pp": "V.PTCP",
    "supn": "V.CVB"
}


def pos_map(p: str):
    """For mapping to coarse POS tags in cases where some 
    languages have a bit more granularity"""

    # TODO: Add exact mappings as we come across them
    MAP = {

    }

    # Icelandic vblex, etc.
    if p.startswith("vb") or p == "vaux":
        return "verb"

    return MAP.get(p, p)


def read_apt(fn: str):
    lex = {}
    with open(fn, "r") as f:
        for line in f:
            word, *tags = line.rstrip().split("\t")
            lex.setdefault(word, set())
            if tags:
                [lex[word].add(t) for t in tags[0].split(",")]

    return lex


def read_tumpc_map(fn):
    m = {}
    with open(fn, "r") as f:
        for line in f:
            tumpc_slot, uni_tag = line.rstrip().split("\t")
            m[tumpc_slot] = uni_tag

    return m


class BaseClassifier:
    """Base class for reading an analysis, classifying its errors, and formatting its tags"""

    # TODO: Finnish (fin)
    lang2code = {
        "deu": "de_DE",
        "isl": "is",
        "rus": "ru_RU",
        "swe": "sv_SE",
    }

    def __init__(self, lang: str, wiki_lex_errors: Set, apt_lex: Dict):
        self.lang = lang
        self.wiki_lex_errors = wiki_lex_errors
        self.apt_lex = apt_lex
        self.spellchecker = phunspell.Phunspell(self.lang2code[lang])
        self.last_tags = ({}, {})
        self.last_poses = ({}, {})
        self.valid_pos = valid_pos_fac[lang]

    def classify_error(
        self,
        row: List,
        src_analyses: List,
        tgt_analyses: List
    ) -> List:
        """Automatically detect one of the 4 errors in stage I of our pipeline
        from the analysis from apertium"""
        errors = []
        src_word, tgt_word, src_slot, src_msd, tgt_slot, tgt_msd = row

        # 1. UNK analysis and LEXICAL ERRORs
        if self._is_unk(src_analyses):
            if self._lex_error(src_word):
                errors.append(SRC_LEXICAL_ERROR)
            else:
                errors.append("UNK")
                return errors

        if self._is_unk(tgt_analyses):
            if self._lex_error(tgt_word):
                errors.append(TGT_LEXICAL_ERROR)
            else:
                errors.append("UNK")
                return errors

        # [(lem, pos, tags)]
        s_anns = self.get_all_analyses(src_analyses)
        t_anns = self.get_all_analyses(tgt_analyses)

        if any(errors):
            self.last_tags = ({}, {})
            self.last_poses = ({}, {})
            errors.extend(self.classify_slot_error(
                row,
                s_anns,
                errors
            ))
            return errors

        # This is stored for later. `Last` refers to the last sample analyzed
        self.last_tags = ([s[-1] for s in s_anns], [t[-1] for t in t_anns])
        self.last_poses = ([s[1] for s in s_anns], [t[1] for t in t_anns])

        # For finding intersection
        sposes = set(self.last_poses[0])
        tposes = set(self.last_poses[1])

        # 2. POS PAIR ERROR
        pos_intsct = sposes & tposes
        # FIXME: We remove the hierarchy here. This means that
        # POS_ERROR and POS_PAIR_ERROR can co-occur.
        if len(pos_intsct) < 1:
            # return [POS_PAIR_ERROR]
            errors.append(POS_PAIR_ERROR)

        # Filter lemmas to those of the same POS
        slemmas = set([s for s, p, _ in s_anns if p in pos_intsct])
        tlemmas = set([t for t, p, _ in t_anns if p in pos_intsct])

        # 3. POS ERRORs
        #    for now, we consider only adj, noun, verb for now
        if len(sposes & self.valid_pos) < 1:
            errors.append(SRC_POS_ERROR)
        if len(tposes & self.valid_pos) < 1:
            errors.append(TGT_POS_ERROR)

        if any(errors):
            errors.extend(self.classify_slot_error(
                row,
                s_anns,
                errors
            ))
            return errors

        # 3. PARADIGM_ERROR
        if len(slemmas & tlemmas) < 1:
            errors = [PARADIGM_ERROR]
            errors.extend(self.classify_slot_error(
                row,
                s_anns,
                errors
            ))
            return errors

        errors = [CORRECT]
        errors.extend(self.classify_slot_error(
            row,
            s_anns,
            errors
        ))
        return errors

    def classify_slot_error(
        self,
        row: List,
        src_anns: List,
        errors: List
    ) -> List:
        """Automatically detect one of the 4 errors in stage I of our pipeline
        from the analysis from apertium"""
        src_word, tgt_word, _, src_msd, _, tgt_msd = row
        slot_errors = []
        src_gold = self.apt_lex.get(src_word)

        # This should not happen but catches anything we missed when mapping
        if not src_gold:
            slot_errors.append(MANUALLY_CHECK_SRC_SLOT_ERROR)
            print(MANUALLY_CHECK_SRC_SLOT_ERROR, src_word, src_gold)
        elif len(src_gold & INVALID_APT_TAGS) == len(src_gold):
            # Get the single string from the set
            (src_gold,) = src_gold
            # Mark that we have an ERR in the mapping
            slot_errors.append(f"{src_gold}_SRC_SLOT_ERROR")
        elif src_msd not in src_gold:
            slot_errors.append(SRC_SLOT_ERROR)

        tgt_gold = self.apt_lex.get(tgt_word)
        # This should not happen but catches anything we missed when mapping
        if not tgt_gold:
            slot_errors.append(MANUALLY_CHECK_TGT_SLOT_ERROR)
            print(MANUALLY_CHECK_TGT_SLOT_ERROR, src_word, src_gold)
        elif len(tgt_gold  & INVALID_APT_TAGS) == len(tgt_gold):
            # Get the single string from the set
            (tgt_gold,) = tgt_gold
            # Mark that we have an ERR in the mapping
            slot_errors.append(f"{tgt_gold}_TGT_SLOT_ERROR")
        elif tgt_msd not in tgt_gold:
            slot_errors.append(TGT_SLOT_ERROR)
        # If correct, check that the mapped tags are of the same POS
        # Otherwise, its an error
        if (
            errors == [CORRECT] and
            not slot_errors and
            src_msd.split(";")[0] != tgt_msd.split(";")[0]
        ):
            # Now that one of several possible MSDs for a word has been asserted in the training data,
            # We can end up with samples where the target analysis chosen is actually from
            # a POS that the src form cannot be. These are also POS_PAIR_ERRORS
            # NOTE that previously caught POS_PAIR_ERRORS occur when there is no possible analysis s.t.
            #      they share a POS. Here, we analyze the particular analysis asserted by the MSD
            if tgt_msd.split(";")[0] not in [APERT_POS_TO_UNI.get(a[1], "") for a in src_anns]:
                slot_errors.append(POS_PAIR_ERROR)
            else:
                # Seperately, a MAPPED_POS_PAIR_ERROR occurs when the assigned MSDs are of different POS,
                # but the chosen tgt MSD COULD share a paradigm with the src.
                # Because in practice we ignore the src MSD for training, 
                # these should never be a problem, though.
                slot_errors.append(MAPPED_POS_PAIR_ERROR)

        return slot_errors

    def get_last_tags(self) -> str:
        # These get cached each time we classify the error on the analysis.
        # TODO: this is janky,  since it relies on classify_error
        # having already been called -- we can read the analysis just once more elegeantly.
        src, tgt = self.last_tags
        src_p, tgt_p = self.last_poses

        def fs(p, t):
            return ";".join([p, t])

        src = " - ".join(map(fs, src_p, src))
        tgt = " - ".join(map(fs, tgt_p, tgt))

        return src, tgt

    def get_all_analyses(self, analyses: List):
        """Read each analysis string and produce sets of all pieces
        
        return: List[Tuple]"""
        ann_list = []
        for a in analyses:
            # For HFST the analysis has the surface form in the first list index
            if isinstance(a, list):
                a = a[1]

            ann_list.append(self.read_analysis(a))

        return ann_list

    def read_analysis(self, analysis: str):
        """Read the string of a single analysis
        
        return: (lemma: str, pos: str, tags: str)"""
        lemma, *tags = analysis.rstrip("\n").split(";")
        pos, morphtags = "", ""
        if tags:
            pos = tags.pop(0)
            morphtags = ";".join(tags)

        return lemma, pos_map(pos), morphtags


class LtClassifier(BaseClassifier):
    """Reads the analysis from the binary FSTs (called with lt-proc)"""

    def _is_unk(self, a):
        return not a or a[0].startswith("*")

    def _lex_error(self, word: str):
        """Check if the word is not in wikipedia and not in the spellchecker
        
        If either resource recongizes it, it is not a lex_error"""
        return word in self.wiki_lex_errors and not self.spellchecker.lookup(word)

    def _read_analyses(self, a: str):
        inp, *analyses = a.split("/")

        return analyses


def postprocess_annotations(err: List):
    # Finalize the errors
    errors_str = ";".join(err)
    # Do not mark it as UNK if we have a LEXICAL_ERROR
    # The UNK slot errors are also overridden by lexical errors
    if "LEXICAL" in errors_str:
        err = [e for e in err if not e.startswith("UNK")]
    # Correct annotations can have slot/mapped_pos_pair errors
    # added to them at a later stage, making them no longer correct
    if len(err) > 1 and CORRECT in err:
        err.remove(CORRECT)
    # TODO: ERR slot errors? I think they are fine as they are
    return err


def make_row(row, src_tags, tgt_tags, err) -> List[List]:
    err = postprocess_annotations(err)
    row.extend([src_tags, tgt_tags, " ".join(err)])

    return row


def read_csv(fn: str) -> Generator:
    rows = []
    with open(fn, "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        # Skip header row
        next(reader, None)
        for row in reader:
            if any(row):
                rows.append(row)

    return rows


def get_apt_analyses(fn):
    anlzr = {}
    with open(fn, "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, anlses = line.split("\t")
                anlzr.setdefault(word, []).extend(anlses.split(","))

    return anlzr


def read_wiki_lex_errors(filename: str):
    words = set()
    with open(filename, "r") as f:
        for line in f:
            words.add(line.rstrip())

    return words


def add_msds_to_row(row: List, tumpc_map: Dict):
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

    return [src_word, tgt_word, src_slot, src_msd, tgt_slot, tgt_msd]


@click.command()
@click.option("--anns_fn", required=True, help="Path to a csv with the data")
@click.option("--apertium_fn", required=True, help="Path to an apertium lexicon file")
@click.option("--apertium_mapped_fn", required=True, type=str)
@click.option("--tumpc_map_fn", required=True, type=str)
@click.option(
    "--wiki_lexerrors_fn", required=True,
    help="Path to file of UNK words that do not exist in wikipedia"
)
@click.option("--outfn", required=True)
@click.option("--lang", required=True, help="3 letter ISO code")
def main(anns_fn, apertium_fn, apertium_mapped_fn, tumpc_map_fn, wiki_lexerrors_fn, outfn, lang):
    # Each row has the csv fields:
    # src tgt src_slot src_predicted msd tgt_slot tgt_predicted msd annotation notes
    new_rows = []
    rows = read_csv(anns_fn)
    apt_lex = read_apt(apertium_mapped_fn)
    tumpc_map = read_tumpc_map(tumpc_map_fn)
    wiki_lex_errors = read_wiki_lex_errors(wiki_lexerrors_fn)
    rows = [add_msds_to_row(row, tumpc_map) for row in rows]
    classifier = LtClassifier(lang, wiki_lex_errors=wiki_lex_errors, apt_lex=apt_lex)
    analyzer = get_apt_analyses(apertium_fn)

    print("Analyzing src words...")
    all_src_analyses = [analyzer[r[0]] for r in rows]
    print("Analyzing tgt words...")
    all_tgt_analyses = [analyzer[r[1]] for r in rows]
    print("Classifying errors")
    for row, src_analysis, tgt_analysis in tqdm(list(zip(rows, all_src_analyses, all_tgt_analyses))):
        err = classifier.classify_error(row, src_analysis, tgt_analysis)
        src_tags, tgt_tags = classifier.get_last_tags()

        new_rows.append(make_row(row, src_tags, tgt_tags, err))

    HEADER = "src word,tgt word,src slot,src msd,tgt slot,tgt msd,src apertium tags,tgt apertium tags,annotation"
    # Write the csv of new rows
    with open(outfn, "w") as o:
        o.write(HEADER + "\n")
        o.write("\n".join([",".join(r) for r in new_rows]))


if __name__=='__main__':
    main()
    # run_test()