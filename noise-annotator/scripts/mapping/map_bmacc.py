import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching

import click
from collections import defaultdict

# Based on 2020 SIGMORPHON shared task on Unsupervised Morphological Paradigm Completion


INVALID_TAGS = set(["UNK", "ERR"])


def read_lexicon(fn):
    """
    :param path: path to file of a tagged lexicon
    :returns: Dict of set like {tag: {word1, ..., wordn}}.
    """
    d = defaultdict(set)
    with open(fn, "r") as f:
        for line in f:
            word, *tag = line.rstrip().split("\t")
            if tag and tag[0] not in INVALID_TAGS:
                tag=tag[0]
                d[tag].add(word)

    return d


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

        matched_idxs = {
            p.lstrip("prd_"): r.lstrip("ref_") for p, r in matches.items() if p in prd_nodes
        }
        return matched_idxs


@click.command()
@click.option("--tumpc_fn", required=True, type=str)
@click.option("--gold_fn", required=True, type=str)
@click.option("--outfn", required=True, type=str)
def main(tumpc_fn, gold_fn, outfn):
    # Get dicts of {tag: {word1, ..., wordn}}
    print("Reading gold lexicon...")
    gold_tags = read_lexicon(gold_fn)

    print("Reading tUMPC lexicon...")
    tumpc_tags = read_lexicon(tumpc_fn)

    mapper = Mapper(gold_tags)
    # Get mapping that maximizes micro F1 wrt gold tags
    # TODO: is micro F1 right? I think yes.
    mapping = mapper.map(tumpc_tags)

    # Write mapping
    with open(outfn, "w") as o:
        for k, v in mapping.items():
            o.write(f"{k}\t{v}\n")


if __name__=='__main__':
    main()