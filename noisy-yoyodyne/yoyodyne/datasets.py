"""Dataset classes."""

import csv
import os
import pickle
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import random
import math

import torch
from torch.utils import data

from . import special, util


class Error(Exception):
    """Module-specific exception."""

    pass


class DatasetNoFeatures(data.Dataset):
    """Dataset object.

    The user specifies:

    * an input filename path
    * the 1-based indices for the columns (defaults: source is 1, target is 2)
    * separator characters used to split the input columns strings, with an
      empty string used to indicate that the string should be split into
      Unicode characters

    These together define an enormous set of possibilities; the defaults
    correspond to the SIGMORPHON 2017 data format.
    """

    filename: str
    source_col: int
    target_col: int
    source_sep: str
    target_sep: str
    source_symbol2i: Dict[str, int]
    source_i2symbol: List[str]
    target_symbol2i: Dict[str, int]
    target_i2symbol: List[str]
    no_target: bool

    def __init__(
        self,
        filename,
        tied_vocabulary,
        source_col=1,
        target_col=2,
        source_sep="",
        target_sep="",
        **kwargs,
    ):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
            source_col (int, optional): 1-indexed column in TSV containing
                source strings.
            target_col (int, optional): 1-indexed column in TSV containing
                target strings.
            source_sep (str, optional): separator character between symbols in
                source string. "" treats each character in source as a symbol.
            target_sep (str, optional): separator character between symbols in
                target string. "" treats each character in target as a symbol.
            **kwargs: ignored.
        """
        if source_col < 1:
            raise Error(f"Invalid source column: {source_col}")
        self.source_col = source_col
        if target_col == 0:
            util.log_info("Ignoring targets in input")
        if target_col < 0:
            raise Error(f"Invalid target column: {target_col}")
        self.target_col = target_col
        self.source_sep = source_sep
        self.target_sep = target_sep
        self.samples = list(self._iter_samples(filename))
        self._make_indices(tied_vocabulary)

    @staticmethod
    def _get_cell(row: List[str], col: int, sep: str) -> List[str]:
        """Returns the split cell of a row.

        Args:
           row (List[str]): the split row.
           col (int): the column index
           sep (str): the string to split the column on; if the empty string,
              the column is split into characters instead.

        Returns:
           List[str]: symbols from that cell.
        """
        cell = row[col - 1]  # -1 because we're using one-based indexing.
        return list(cell) if not sep else cell.split(sep)

    def _iter_samples(
        self, filename: str
    ) -> Iterator[Tuple[List[str], Optional[List[str]]]]:
        """Yields specific input samples from a file.

        Args:
            filename (str): input file.

        Yields:
            Tuple[List[str], Optional[List[str]]]: Tuple
                of source and target string. (Target string
                is None if self.target_col is 0).
        """
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                target = (
                    self._get_cell(row, self.target_col, self.target_sep)
                    if self.target_col
                    else None
                )
                yield source, target

    def _make_indices(self, tied_vocabulary: bool) -> None:
        """Generates Dicts for encoding/decoding symbols as unique indices.

        Args:
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
        """
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        for source, target in self.samples:
            source_vocabulary.update(source)
            # Only updates if target.
            if self.target_col:
                target_vocabulary.update(target)
        if tied_vocabulary:
            source_vocabulary.update(target_vocabulary)
            if self.target_col:
                target_vocabulary.update(source_vocabulary)
        self.source_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(source_vocabulary)
            )
        }
        self.source_i2symbol = list(self.source_symbol2i.keys())
        self.target_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(target_vocabulary)
            )
        }
        self.target_i2symbol = list(self.target_symbol2i.keys())

    @staticmethod
    def _write_pkl(obj: Any, path: str) -> None:
        """Writes pickled object to path.

        Args:
            obj (Any): the object to be written.
            path (str): output path.
        """
        with open(path, "wb") as sink:
            pickle.dump(obj, sink)

    @staticmethod
    def _read_pkl(path: str) -> Any:
        """Reads a pickled object from the path.

        Args:
            path (str): input path.

        Returns:
            Any: the object read.
        """
        with open(path, "rb") as source:
            return pickle.load(source)

    def write_index(self, outdir: str, filename: str) -> None:
        """Saves character mappings.

        Args:
            outdir (str): output directory.
            filename (str): output filename.
        """
        vocab = {
            "source_symbol2i": self.source_symbol2i,
            "source_i2symbol": self.source_i2symbol,
            "target_symbol2i": self.target_symbol2i,
            "target_i2symbol": self.target_i2symbol,
        }
        self._write_pkl(
            vocab,
            os.path.join(outdir, f"{filename}_vocab.pkl"),
        )

    def load_index(self, indir: str, filename: str) -> None:
        """Loads character mappings.

        Args:
            indir (str): input directory.
            filename (str): input filename.
        """
        vocab = self._read_pkl(os.path.join(indir, f"{filename}_vocab.pkl"))
        self.source_symbol2i = vocab["source_symbol2i"]
        self.source_i2symbol = vocab["source_i2symbol"]
        self.target_symbol2i = vocab["target_symbol2i"]
        self.target_i2symbol = vocab["target_i2symbol"]

    def encode(
        self,
        symbol2i: Dict,
        word: List[str],
        add_start_tag: bool = True,
        add_end_tag: bool = True,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with word boundary IDs.

        Args:
            symbol2i (Dict).
            word (List[str]): word to be encoded.
            unk (int): default idx to return if symbol is outside symbol2i.
            add_start_tag (bool, optional): whether the sequence should be
                prepended with a start tag.
            add_end_tag (bool, optional): whether the sequence should be
                prepended with a end tag.

        Returns:
            torch.Tensor: the encoded tensor.
        """
        sequence = []
        if add_start_tag:
            sequence.append(special.START)
        sequence.extend(word)
        if add_end_tag:
            sequence.append(special.END)
        return torch.LongTensor(
            [symbol2i.get(symbol, self.unk_idx) for symbol in sequence]
        )

    def _decode(
        self,
        indices: torch.Tensor,
        decoder: List[str],
        symbols: bool,
        special: bool,
    ) -> List[List[str]]:
        """Decodes the tensor of indices into characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            decoder (List[str]): decoding lookup table.
            symbols (bool): whether to include the regular symbols when
                decoding the string.
            special (bool): whether to include the special symbols when
                decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """

        def include(c: int) -> bool:
            """Whether to include the symbol when decoding.

            Args:
                c (int): a single symbol index.

            Returns:
                bool: if True, include the symbol.
            """
            include = False
            is_special_char = c in self.special_idx
            if special:
                include |= is_special_char
            if symbols:
                # Symbols will be anything that is not SPECIAL.
                include |= not is_special_char
            return include

        decoded = []
        for index in indices.cpu().numpy():
            decoded.append([decoder[c] for c in index if include(c)])
        return decoded

    def decode_source(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of source indices, returns a list of characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                alphabet when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            indices,
            decoder=self.source_i2symbol,
            symbols=symbols,
            special=special,
        )

    def decode_target(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of target indices, returns a list of characters.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                alphabet when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        return self._decode(
            indices,
            decoder=self.target_i2symbol,
            symbols=symbols,
            special=special,
        )

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_symbol2i)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_symbol2i)

    @property
    def pad_idx(self) -> int:
        return self.source_symbol2i[special.PAD]

    @property
    def start_idx(self) -> int:
        return self.source_symbol2i[special.START]

    @property
    def end_idx(self) -> int:
        return self.source_symbol2i[special.END]

    @property
    def unk_idx(self) -> int:
        return self.source_symbol2i[special.UNK]
    
    @property
    def mask_idx(self) -> int:
        return self.source_symbol2i[special.MASK]

    @property
    def special_idx(self) -> Set[int]:
        """The set of indexes for all `special` symbols."""
        return {self.unk_idx, self.pad_idx, self.start_idx, self.end_idx, self.mask_idx}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: source/target sample to be
                consumed by the model.
        """
        source, target = self.samples[idx]
        source_encoded = self.encode(self.source_symbol2i, source)
        target_encoded = (
            self.encode(self.target_symbol2i, target, add_start_tag=False)
            if self.target_col
            else None
        )
        return source_encoded, target_encoded


class DatasetFeatures(DatasetNoFeatures):
    """Dataset object with separate features.

    This accepts an additional secondary input of feature labels. Features are
    specified in a similar way to source and target.

    The user specifies:

    * an input filename path
    * the 1-based indices for the columns (defaults: source is 1,
      target is 2, features is 3)
    * separator characters used to split the input columns strings, with an
      empty string used to indicate that the string should be split into
      Unicode characters

    These together define an enormous set of possibilities; the defaults
    correspond to the SIGMORPHON 2017 data format.
    """

    features_col: int
    features_sep: str
    features_idx: int

    def __init__(self, *args, features_col=3, features_sep=";", **kwargs):
        """Initializes the dataset.

        Args:
            filename (str): input filename.
            tied_vocabulary (bool): whether or not to share the
                source/target vocabularies.
            source_col (int optional): 1-indexed column in TSV containing
                source strings.
            target_col (int, optional): 1-indexed column in TSV containing
                target strings.
            source_sep (str, optional): separator character between symbols in
                source string. "" treats each character in source as a symbol.
            target_sep (str, optional): separator character between symbols in
                target string. "" treats each character in target as symbol.
            features_col (int, optional): 1-indexed column in TSV containing
                features labels.
            features_sep (str, optional): separator character between symbols
                in target string. "" treats each character in target as symbol.
            **kwargs: passed to superclass constructor.
        """
        if features_col < 0:
            raise Error(f"Invalid features column: {features_col}")
        util.log_info("Including features")
        self.features_col = features_col
        self.features_sep = features_sep
        self.features_idx = 0
        super().__init__(*args, **kwargs)

    def _iter_samples(
        self, filename: str
    ) -> Iterator[Tuple[List[str], List[str], Optional[List[str]]]]:
        """Yields specific input samples from a file.

        Sames as in superclass, but also handles features.

        Args:
            filename (str): input file.

        Yields:
            Tuple[List[str], List[str], Optional[List[str]]]:
                Source, Features, Target tuple. (Target is None
                if self.target_col is 0).
        """
        with open(filename, "r") as source:
            tsv_reader = csv.reader(source, delimiter="\t")
            for row in tsv_reader:
                source = self._get_cell(row, self.source_col, self.source_sep)
                if len(row) >= self.features_col:
                    features = self._get_cell(
                        row, self.features_col, self.features_sep
                    )
                    # Use unique encoding for features.
                    # This disambiguates from overlap with source.
                    features = [f"[{feature}]" for feature in features]
                else:
                    features = []
                target = (
                    self._get_cell(row, self.target_col, self.target_sep)
                    if self.target_col
                    else None
                )
                yield source, features, target

    def _make_indices(self, tied_vocabulary: bool) -> None:
        """Generates unique indices dictionaries.

        Same as in superclass, but also handles features.

        Args:
            tied_vocabulary (bool): whether the source and target should
                share a vocabulary.
        """
        # Ensures the idx of special symbols are identical in both vocabs.
        special_vocabulary = special.SPECIAL.copy()
        target_vocabulary: Set[str] = set()
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        for source, features, target in self.samples:
            source_vocabulary.update(source)
            features_vocabulary.update(features)
            # Only updates if target.
            if self.target_col:
                target_vocabulary.update(target)
        if tied_vocabulary:
            source_vocabulary.update(target_vocabulary)
            if self.target_col:
                target_vocabulary.update(source_vocabulary)
        source_vocabulary = special_vocabulary + sorted(source_vocabulary)
        # Source and features vocab share embedding dict.
        # features_idx assists in indexing features.
        self.features_idx = len(source_vocabulary)
        self.source_symbol2i = {
            c: i
            for i, c in enumerate(
                source_vocabulary + sorted(features_vocabulary)
            )
        }
        self.source_i2symbol = list(self.source_symbol2i.keys())
        self.target_symbol2i = {
            c: i
            for i, c in enumerate(
                special_vocabulary + sorted(target_vocabulary)
            )
        }
        self.target_i2symbol = list(self.target_symbol2i.keys())

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                source/features/target sample to be consumed by the model.
        """
        source, features, target = self.samples[idx]
        source_encoded = self.encode(self.source_symbol2i, source)
        features_encoded = self.encode(
            self.source_symbol2i,
            features,
            add_start_tag=False,
            add_end_tag=False,
        )
        target_encoded = (
            self.encode(self.target_symbol2i, target, add_start_tag=False)
            if self.target_col
            else None
        )
        return source_encoded, features_encoded, target_encoded

    def decode_source(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of source indices, returns a list of characters.

        Overriding to prevent use of features encoding.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        # Masking features vocab.
        indices = torch.where(
            indices < self.features_idx, indices, self.pad_idx
        )
        return self._decode(
            indices,
            decoder=self.source_i2symbol,
            symbols=symbols,
            special=special,
        )

    def decode_features(
        self,
        indices: torch.Tensor,
        symbols: bool = True,
        special: bool = True,
    ) -> List[List[str]]:
        """Given a tensor of feature indices, returns a list of characters.

        This is simply an alias for using decode_source for features that
        manages the use of a separate SPECIAL vocabulary for features.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbols (bool, optional): whether to include the regular symbols
                when decoding the string.
            special (bool, optional): whether to include the special symbols
                when decoding the string.

        Returns:
            List[List[str]]: decoded symbols.
        """
        # Masking source vocab.
        indices = torch.where(
            (indices >= self.features_idx) | (indices < len(self.special_idx)),
            indices,
            self.pad_idx,
        )
        return self._decode(
            indices,
            decoder=self.source_i2symbol,
            symbols=symbols,
            special=special,
        )

    @property
    def features_vocab_size(self) -> int:
        return len(self.source_symbol2i) - self.features_idx

    def write_index(self, outdir: str, filename: str) -> None:
        # Overwrites method to save features encoding.
        vocab = {
            "source_symbol2i": self.source_symbol2i,
            "source_i2symbol": self.source_i2symbol,
            "target_symbol2i": self.target_symbol2i,
            "target_i2symbol": self.target_i2symbol,
            "features_idx": self.features_idx,
        }
        self._write_pkl(
            vocab,
            os.path.join(outdir, f"{filename}_vocab.pkl"),
        )

    def load_index(self, indir: str, filename: str) -> None:
        # Overwrites method to load features encoding.
        vocab = self._read_pkl(os.path.join(indir, f"{filename}_vocab.pkl"))
        self.source_symbol2i = vocab["source_symbol2i"]
        self.source_i2symbol = vocab["source_i2symbol"]
        self.target_symbol2i = vocab["target_symbol2i"]
        self.target_i2symbol = vocab["target_i2symbol"]
        self.features_idx = vocab["features_idx"]


# DEPRECATED IN FAVOR OF MASKIN IN COLLATOR
# class PretrainingDataset(DatasetNoFeatures):
#     """Dataset object for pretraining.

#     The user specifies:

#     * an input filename path
#     * the 1-based indices for the columns (defaults: source is 1, target is 2)
#     * separator characters used to split the input columns strings, with an
#       empty string used to indicate that the string should be split into
#       Unicode characters

#     These together define an enormous set of possibilities; the defaults
#     correspond to the SIGMORPHON 2017 data format.

#     At runtime, each input pair has a mask applied to it such that 20% of the inputs masked as in BERT/RoBERTa (but we use 20% instead of 15 to account for the fact that many words are shorter sequences):
#     1. 80% of the time a special [MASK] symbol.
#     2. 10% of the time, we leave the character as is.
#     3. 10% of the time, the character is replaced by a random (non-special) character from the input vocab.

#     """

#     MASK_PROPORITON: float = 0.20
#     MASK_PROB: float = 0.80
#     IDENTITY_PROB: float = 0.10
#     NEW_VOCAB_PROB: float = 0.10

#     def __init__(self, *args, do_masking=True, **kwargs):
#         """Initializes the Pretraining dataset."""
#         self.do_masking = do_masking
#         super().__init__(*args, **kwargs)
#         self.VALID_VOCAB = [
#             v for i, v in enumerate(self.source_i2symbol) if i not in self.special_idx
#         ]

#     def _get_mask(self, char: str):
#         return special.MASK

#     def _get_identity(self, char: str):
#         return char

#     def _get_random_vocab(self, char: str):
#         return random.sample(self.VALID_VOCAB, 1)[0]

#     def _sample_replacement(self, char: str) -> str:

#         SAMPLER = {
#             1: self._get_mask,
#             2: self._get_identity,
#             3: self._get_random_vocab,
#         }

#         choice = random.choices(
#             [1, 2, 3],
#             weights = [self.MASK_PROB, self.IDENTITY_PROB, self.NEW_VOCAB_PROB]
#         )[0]

#         return SAMPLER[choice](char)

#     def _apply_mask(self, word: str) -> str:
#         indices = [i for i in range(len(word))]
#         num_samples = math.ceil(self.MASK_PROPORITON * len(word))
#         indices = random.sample(indices, num_samples)
#         for i in indices:
#             word[i] = self._sample_replacement(word[i])

#         return word

#     def __getitem__(
#         self, idx: int
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """Retrieves item by index.

#         Args:
#             idx (int).

#         Returns:
#             Tuple[torch.Tensor, torch.Tensor]: source/target sample to be
#                 consumed by the model.
#         """
#         source, target = self.samples[idx]
#         if self.do_masking:
#             source = self._apply_mask(source)
#         source_encoded = self.encode(self.source_symbol2i, source)
#         target_encoded = (
#             self.encode(self.target_symbol2i, target, add_start_tag=False)
#             if self.target_col
#             else None
#         )

#         return source_encoded, target_encoded


def get_dataset_cls(include_features: bool) -> torch.utils.data.Dataset:
    """Dataset factory.

    Args:
        arch (str): the string label for the architecture.
        include_features (bool).

    Returns:
        data.Dataset: the desired dataset class.
    """
    return DatasetFeatures if include_features else DatasetNoFeatures
