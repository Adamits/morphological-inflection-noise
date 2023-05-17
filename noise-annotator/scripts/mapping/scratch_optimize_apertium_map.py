import click
import subprocess
import random
from typing import List

import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching


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
    pairs = []
    with open(fn, "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                _, word, msd = line.split("\t")
                pairs.append((word, msd))

    return pairs


def format_analysis(analyses: List):
    """Format the apertium analysis into comma delimited string of ;-delimited tags"""
    frmtd_analysis = []
    # TODO: Handle '$'
    for a in analyses:
        a = a.rstrip("\n").split("$")[0]
        a = a.split("#")[0]
        _, *tags = a.replace(">", "").split("<")
        frmtd_analysis.append(";".join(tags))
    
    return frmtd_analysis


def parse_hfst_analysis(a: str):
    lines = a.split("\n")
    analyses = []
    # For tracking special case that is unlikely
    compound_analyses = []
    for l in lines:
        l = l.rstrip("\n").lstrip("> ").rstrip(">")
        if not l:
            continue
        
        if ">+" in l:
            compound_analyses.append(l.split("\t"))
        else:
            analyses.append(l.split("\t"))

    # If all analyses are compound, then we keep all
    if not analyses:
        analyses = [c[1] for c in compound_analyses if c]
    # Otherwise we filter out the compound analyses as they are very unlikely
    # and a bit confusing for lemma analysis.
    else:
        analyses = [a[1] for a in analyses if a]

    return format_analysis(analyses)


def parse_lt_analysis(aprt: str):
    inp, *analyses = aprt.rstrip().split("/")

    frmtd_analysis = format_analysis(analyses)

    return frmtd_analysis


def run_analyzer(analyzer_path: str, word: str) -> str:
    """Run the apertium analyzer, return the string of analysis"""
    process = "hfst-lookup" if analyzer_path.endswith(".hfst") else "lt-proc"
    cmd = f"echo {word} | {process} {analyzer_path}"

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    output, error = process.communicate()

    if error is not None:
        raise Exception(error)

    parse_func = parse_hfst_analysis if analyzer_path.endswith(".hfst") else parse_lt_analysis

    return parse_func((output).decode("utf-8"))

class Mapper:
    def __init__(self, reference):
        """
        :param reference: The gold reference. Dict of dict of set like {tag: {word1, ..., wordn}}.
        """
        self._check_dict_dict(reference, lambda s: isinstance(s, set) and len(s) > 0, 'reference')
        self.reference = reference

    def _check_dict_dict(self, dd, check_value=None, arg_name=''):
        if not isinstance(dd, dict) or not all(isinstance(key, str) for key in dd.keys()) \
                or not all(isinstance(value, set) for value in dd.values()):
            raise ValueError("'{}' is not of type dict[str, set]".format(arg_name))

    def _get_metric(self, true_pos, prd_size, ref_size, metric):
        precision = true_pos / prd_size if prd_size > 0 else 0
        recall = true_pos / ref_size if ref_size > 0 else 0
        if metric == 'precision':
            return precision
        elif metric == 'recall':
            return recall
        elif metric == 'f1':
            return precision * recall * 2 / (precision + recall) if precision + recall > 0 else 0

    def map(self, prediction, metric='f1', average='micro'):
        """
        :param prediction: The prediction of your model. Dict of dict like {tag: {lemma: word}}.
        :param metric: Metric for calculating per-slot-pair score.
        :param average: Type of average.
        :returns: The overall score.
        """
        self._check_dict_dict(prediction, lambda s: isinstance(s, str), 'prediction')
        if metric not in ['f1', 'precision', 'recall']:
            raise ValueError("'metric' must be 'precision', 'recall' or 'f1', got '{}'.".format(metric))
        if average not in ['micro', 'macro']:
            raise ValueError("'average' must be 'micro' or 'macro', got '{}'.".format(average))

        graph = nx.Graph()
        prd_nodes = frozenset('prd_' + tag for tag in prediction.keys())
        ref_nodes = frozenset('ref_' + tag for tag in self.reference.keys())
        graph.add_nodes_from(prd_nodes)
        graph.add_nodes_from(ref_nodes)

        for prd_tag, prd_set in prediction.items():
            for ref_tag, ref_set in self.reference.items():
                true_pos = len(prd_set & ref_set)
                if average == 'micro':
                    tag_score = true_pos
                else:
                    tag_score = self._get_metric(true_pos, len(prd_set), len(ref_set), metric)

                graph.add_edge('prd_' + prd_tag, 'ref_' + ref_tag, weight=-tag_score)

        matches = minimum_weight_full_matching(graph, prd_nodes)
        # overlap = len(set([p for p, _ in matches.items()]) & set(prd_nodes)) / len(set(prd_nodes))

        matched_idxs = {p.lstrip("prd_"): r.lstrip("ref_") for p, r in matches.items() if p in prd_nodes}
        return matched_idxs


@click.command()
@click.option("--map_fn", required=True)
@click.option("--uni_fn", required=True)
@click.option("--apertium_fn", required=True)
def main(map_fn, uni_fn, apertium_fn):
    # 1. read map
    map, order = read_map(map_fn)
    # 2. Read UniMorph
    uni = read_uni(uni_fn)
    # 3. Run random sample of 200 words through apertium
    pairs_200 = random.sample(uni, 200)
    # 4. Get the corresponding apertium analyses
    print("Analyzing Unimorph words...")
    apt_analyses = [run_analyzer(apertium_fn, word) for (word, msd) in pairs_200]
    print("Computing maps...")
    #apt_analyses = [set([apply_map(a, map, order) for a in anlsis]) for anlsis in apt_analyses]
    apt_tags = {}
    uni_tags = {}
    uni_words = {}
    for (word, msd), apt_msds in zip(pairs_200, apt_analyses):
        apt_msds = [a for a in apt_msds if a]
        for apt in apt_msds:
            apt_tags.setdefault(apt, set()).add(word)

        uni_tags.setdefault(msd, set()).add(word)
        uni_words.setdefault(word, set()).add(msd)

    print("Optimizing match...")
    counts = {}
    for t, words in apt_tags.items():
        counts.setdefault(t, {})
        for w in words:
            for unitag in uni_words[w]:
                counts[t].setdefault(unitag, 0)
                counts[t][unitag] += 1

    mapping = {}
    for t in apt_tags:
        if counts[t] == 0:
            print(f"No matches for {t}")
            continue
        mapping[t] = max(counts[t], key=counts[t].get)
    # mapper = Mapper(uni_tags)
    # mapping = mapper.map(apt_tags)

    print(mapping)
    for k, v in mapping.items():
        print(k, v)

if __name__=='__main__':
    main()