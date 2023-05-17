"""Collators."""

from typing import Iterable, List, Tuple

import torch
from torch.nn import functional


class Collator:
    """Base class for other collators.

    Pads according to the longest sequence in a batch of sequences."""

    pad_idx: int

    def __init__(self, pad_idx):
        """Initializes the collator.

        Args:
            pad_idx (int).
        """
        self.pad_idx = pad_idx

    @staticmethod
    def max_len(batch: torch.Tensor) -> int:
        """Computes max length for a list of tensors.

        Args:
            batch (List[, torch.Tensor, torch.Tensor]).

        Returns:
            int.
        """
        return max(len(tensor) for tensor in batch)

    @staticmethod
    def concat_tuple(
        b1: Iterable[torch.Tensor], b2: Iterable[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        """Concatenates all tensors in b1 to respective tensor in b2.

        For joining source and feature tensors in batches.

        Args:
            b1 (Iterable[torch.Tensor]): Iterable of tensors.
            b2 (Iterable[torch.Tensor]): Iterable of tensors.

        Returns:
            Tuple[torch.Tensor]: the concatenation of
            parallel entries in b1 and b2.
        """
        return tuple(torch.cat((i, j)) for i, j in zip(b1, b2))

    def pad_collate(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads the batch to the maximum sequence length in the batch.

        Args:
            batch torch.Tensor: A batch of samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        batch = torch.stack(
            [self.pad_tensor(tensor, self.max_len(batch)) for tensor in batch]
        )
        batch_mask = batch == self.pad_idx
        return batch, batch_mask

    def pad_tensor(self, tensor: torch.Tensor, pad_max: int) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor (torch.Tensor).
            pad_max (int): The desired length for the tensor.

        Returns:
            torch.Tensor.
        """
        padding = pad_max - len(tensor)
        return functional.pad(tensor, (0, padding), "constant", self.pad_idx)

    @property
    def has_features(self) -> bool:
        return False

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return False


class SourceCollator(Collator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pads source.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        # Checks if the dataloader passed features.
        if self.is_feature_batch(batch):
            source, features, _ = zip(*batch)
            # Concatenates features with source.
            source = self.concat_tuple(source, features)
        else:
            source, _ = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        return source_padded, source_mask

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return len(batch[0]) == 3


class SourceTargetCollator(SourceCollator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and targets.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        if self.is_feature_batch(batch):
            source, features, target = zip(*batch)
            # Concatenates features with source.
            source = self.concat_tuple(source, features)
        else:
            source, target = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        target_padded, target_mask = self.pad_collate(target)
        return source_padded, source_mask, target_padded, target_mask


class MLMCollator(SourceTargetCollator):

    mask_idx: int
    src_vocab_min_idx: int
    src_vocab_max_idx: int
    mlm_probability: float = 0.2
    MASK_PROB: float = 0.80
    IDENTITY_PROB: float = 0.10
    NEW_VOCAB_PROB: float = 0.10

    def __init__(self, *args, mask_idx, src_vocab_min_idx, src_vocab_max_idx, **kwargs):
        """Initializes the collator.
        """
        self.mask_idx = mask_idx
        self.src_vocab_min_idx = src_vocab_min_idx
        self.src_vocab_max_idx = src_vocab_max_idx
        super().__init__(*args, **kwargs)

    def _is_special(self, idx: int) -> int:
        """Check whether a source symbol idx is a special symbol 
        according to the min/max src vocab. Return 1 if special, 0 otherwise"""
        if idx < self.src_vocab_min_idx or idx > self.src_vocab_max_idx:
            return 1
        else:
            return 0

    def _apply_mask(self, inputs):
        """Based on HF masked language modeling implementation."""
        labels = inputs.clone()
        # Sample a few tokens in each sequence for MLM training.
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            [self._is_special(s) for s in seq] for seq in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.MASK_PROB)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word.
        # We use 0.5 because we have only 20% left of the indices to be replaced.
        # so we take 50% of the 20%.
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            low=self.src_vocab_min_idx,
            high=self.src_vocab_max_idx,
            size=labels.shape,
            dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # The other 10% of tokens stay unchanged.
        return inputs

    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and targets.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        if self.is_feature_batch(batch):
            source, features, target = zip(*batch)
            # Concatenates features with source.
            source = self.concat_tuple(source, features)
        else:
            source, target = zip(*batch)

        source_padded, source_mask = self.pad_collate(source)
        source_padded = self._apply_mask(source_padded)
        target_padded, target_mask = self.pad_collate(target)
        return source_padded, source_mask, target_padded, target_mask


class SourceFeaturesCollator(Collator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and features.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        source, features, _ = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        features_padded, features_mask = self.pad_collate(features)
        return source_padded, source_mask, features_padded, features_mask

    @property
    def has_features(self) -> bool:
        return True

    @staticmethod
    def is_feature_batch(batch: List[torch.Tensor]) -> bool:
        return True


class SourceFeaturesTargetCollator(SourceFeaturesCollator):
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pads source, features, and target.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
            ]
        """
        source, features, target = zip(*batch)
        source_padded, source_mask = self.pad_collate(source)
        target_padded, target_mask = self.pad_collate(target)
        features_padded, features_mask = self.pad_collate(features)
        return (
            source_padded,
            source_mask,
            features_padded,
            features_mask,
            target_padded,
            target_mask,
        )

    @property
    def has_features(self) -> bool:
        return True


class MLMFeaturesCollator(SourceFeaturesTargetCollator):

    mask_idx: int
    src_vocab_min_idx: int
    src_vocab_max_idx: int
    mlm_probability: float = 0.2
    MASK_PROB: float = 0.80
    IDENTITY_PROB: float = 0.10
    NEW_VOCAB_PROB: float = 0.10

    def __init__(self, *args, mask_idx, src_vocab_min_idx, src_vocab_max_idx, **kwargs):
        """Initializes the collator.
        """
        self.mask_idx = mask_idx
        self.src_vocab_min_idx = src_vocab_min_idx
        self.src_vocab_max_idx = src_vocab_max_idx
        super().__init__(*args, **kwargs)

    def _is_special(self, idx: int) -> int:
        """Check whether a source symbol idx is a special symbol 
        according to the min/max src vocab. Return 1 if special, 0 otherwise"""
        if idx < self.src_vocab_min_idx or idx > self.src_vocab_max_idx:
            return 1
        else:
            return 0

    def _apply_mask(self, inputs):
        """Based on HF masked language modeling implementation."""
        labels = inputs.clone()
        # Sample a few tokens in each sequence for MLM training.
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            [self._is_special(s) for s in seq] for seq in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.MASK_PROB)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word.
        # We use 0.5 because we have only 20% left of the indices to be replaced.
        # so we take 50% of the 20%.
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            low=self.src_vocab_min_idx,
            high=self.src_vocab_max_idx,
            size=labels.shape,
            dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # The other 10% of tokens stay unchanged.
        return inputs

    def __call__(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads source and targets.

        Args:
            batch (List[torch.Tensor]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor].
        """
        source, features, target = zip(*batch)

        source_padded, source_mask = self.pad_collate(source)
        source_padded = self._apply_mask(source_padded)
        target_padded, target_mask = self.pad_collate(target)
        # Make single UNK feat if no features
        if not any([f.any() for f in features]):
            features = (torch.ones(source_padded.size(0), 1) * self.mask_idx).long()
        features_padded, features_mask = self.pad_collate(features)
        return (
            source_padded,
            source_mask,
            features_padded,
            features_mask,
            target_padded,
            target_mask,
        )


def get_collator_cls(
    arch: str, include_features: bool, include_targets: bool
) -> Collator:
    """Collator factory.

    Args:
        arch (str).
        include_features (bool).
        include_targets (bool).

    Returns:
        Collator.
    """
    if include_features and arch in ["pointer_generator_lstm", "transducer"]:
        return (
            SourceFeaturesTargetCollator
            if include_targets
            else SourceFeaturesCollator
        )
    else:
        return SourceTargetCollator if include_targets else SourceCollator
