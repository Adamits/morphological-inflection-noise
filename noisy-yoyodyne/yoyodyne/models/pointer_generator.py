"""Pointer-generator model classes."""

from typing import Optional, Tuple

import torch
from torch import nn

from . import attention, base, lstm, generation_probability


class Error(Exception):
    pass


class PointerGeneratorLSTMEncoderDecoderNoFeatures(lstm.LSTMEncoderDecoder):
    """Pointer-generator model with an LSTM backend and no features.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    source_attention: attention.Attention
    geneneration_probability: generation_probability.GenerationProbability

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        # We use the inherited defaults for the source embeddings/encoder.
        enc_size = self.hidden_size * self.num_directions
        self.source_attention = attention.Attention(enc_size, self.hidden_size)
        # Overrides classifier to take larger input.
        self.classifier = nn.Linear(3 * self.hidden_size, self.output_size)
        self.generation_probability = (
            generation_probability.GenerationProbability(
                self.embedding_size, self.hidden_size, enc_size
            )
        )

    def encode(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        encoder: torch.nn.LSTM,
    ) -> torch.Tensor:
        """Encodes the input.

        Args:
            source (torch.Tensor).
            source_mask (torch.Tensor).
            encoder (torch.nn.LSTM).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.source_embeddings(source)
        embedded = self.dropout_layer(embedded)
        lens = (source_mask == 0).sum(dim=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lens, batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim,
        # (D*layers x B x hidden_size, D*layers x B x hidden_size)
        packed_outs, (H, C) = self.encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        H = H.view(
            self.enc_layers, self.num_directions, H.size(1), H.size(2)
        ).sum(axis=1)
        C = C.view(
            self.enc_layers, self.num_directions, C.size(1), C.size(2)
        ).sum(axis=1)
        return encoded, (H, C)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h, last_c = last_hiddens
        source_context, source_attn_weights = self.source_attention(
            last_h.transpose(0, 1), source_enc, source_mask
        )
        _, (h, c) = self.decoder(
            torch.cat((embedded, source_context), 2), (last_h, last_c)
        )
        hidden = h.transpose(0, 1)
        output_probs = self.classifier(
            torch.cat([hidden, source_context], dim=2)
        )
        # Ordinary softmax, log will be taken at the end.
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x output_size.
        attn_probs = torch.zeros(
            symbol.size(0), self.output_size, device=self.device
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x output_size.
        attn_probs.scatter_add_(
            2, source_indices.unsqueeze(1), source_attn_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            source_context, hidden, embedded
        ).unsqueeze(2)
        gen_scores = gen_probs * output_probs
        ptr_scores = (1 - gen_probs) * attn_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)

    def decode(
        self,
        batch_size: int,
        decoder_hiddens: torch.Tensor,
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence.

        Args:
            batch_size (int).
            decoder_hiddens (torch.Tensor).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            target (torch.Tensor, optional).

        Returns:
            torch.Tensor
        """
        # Feeds in the first decoder input, as a start tag.
        # -> B x 1
        decoder_input = (
            torch.LongTensor([self.start_idx])
            .to(self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )
        preds = []
        num_steps = (
            target.size(1) if target is not None else self.max_decode_len
        )
        # Tracks when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size).to(self.device)
        for t in range(num_steps):
            # pred: B x 1 x output_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input,
                decoder_hiddens,
                source_indices,
                source_enc,
                source_mask,
            )
            preds.append(output.squeeze(1))
            # If we have a target (training) then the next input is the gold
            # symbol for this step (teacher forcing).
            if target is not None:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise, it must be inference time and we pass the top pred
            # to the next next timestep (student forcing; greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Tracks which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an END symbol.
                if finished.all():
                    break
        preds = torch.stack(preds)
        return preds

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (base.Batch): tuple of tensors in the batch.

        Returns:
            torch.Tensor.
        """
        # Training mode with targets.
        if len(batch) == 4:
            (source, source_mask, target, target_mask) = batch
        # No targets given at inference.
        elif len(batch) == 2:
            source, source_mask = batch
            target = None
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        batch_size = source.size(0)
        source_encoded, (h_source, c_source) = self.encode(
            source, source_mask, self.encoder
        )
        if self.beam_width is not None and self.beam_width > 1:
            # preds = self.beam_decode(
            #     batch_size, x_mask, enc_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
        else:
            preds = self.decode(
                batch_size,
                (h_source, c_source),
                source,
                source_encoded,
                source_mask,
                target,
            )
        # -> B x output_size x seq_len.
        preds = preds.transpose(0, 1).transpose(1, 2)
        return preds


class PointerGeneratorLSTMEncoderDecoderFeatures(
    PointerGeneratorLSTMEncoderDecoderNoFeatures
):
    """Pointer-generator model with an LSTM backend.

    After:
        See, A., Liu, P. J., and Manning, C. D. 2017. Get to the point:
        summarization with pointer-generator networks. In Proceedings of the
        55th Annual Meeting of the Association for Computational Linguistics
        (Volume 1: Long Papers), pages 1073-1083.
    """

    feature_encoder: nn.LSTM
    linear_h: nn.Linear
    linear_c: nn.Linear
    feature_attention: attention.Attention

    def __init__(self, *args, **kwargs):
        """Initializes the pointer-generator model with an LSTM backend."""
        super().__init__(*args, **kwargs)
        # We use the inherited defaults for the source embeddings/encoder.
        self.feature_encoder = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.enc_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # Initializes the decoder.
        self.linear_h = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear_c = nn.Linear(2 * self.hidden_size, self.hidden_size)
        enc_size = self.hidden_size * self.num_directions
        self.feature_attention = attention.Attention(
            enc_size, self.hidden_size
        )
        # Overrides decoder to be larger.
        self.decoder = nn.LSTM(
            (2 * enc_size) + self.embedding_size,
            self.hidden_size,
            dropout=self.dropout,
            num_layers=self.dec_layers,
            batch_first=True,
        )
        # Overrides classifier to take larger input.
        self.classifier = nn.Linear(5 * self.hidden_size, self.output_size)
        # Overrides GenerationProbability to have larger hidden_size.
        self.generation_probability = (
            generation_probability.GenerationProbability(
                self.embedding_size, self.hidden_size, 2 * enc_size
            )
        )

    def encode(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        encoder: torch.nn.LSTM,
    ) -> torch.Tensor:
        """Encodes the input with the TransformerEncoder.

        Args:
            source (torch.Tensor).
            source_mask (torch.Tensor).
            encoder (torch.nn.LSTM).

        Returns:
            torch.Tensor: sequence of encoded symbols.
        """
        embedded = self.source_embeddings(source)
        embedded = self.dropout_layer(embedded)
        lens = (source_mask == 0).sum(dim=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lens, batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim,
        # (D*layers x B x hidden_size, D*layers x B x hidden_size).
        packed_outs, (H, C) = self.encoder(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Sums over directions, keeping layers.
        # -> num_layers x B x hidden_size.
        H = H.view(
            self.enc_layers, self.num_directions, H.size(1), H.size(2)
        ).sum(axis=1)
        C = C.view(
            self.enc_layers, self.num_directions, C.size(1), C.size(2)
        ).sum(axis=1)
        return encoded, (H, C)

    def decode_step(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        feature_enc: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single step of the decoder.

        This predicts a distribution for one symbol.

        Args:
            symbol (torch.Tensor).
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            feature_enc (torch.Tensor).
            feature_mask (torch.Tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor].
        """
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # -> 1 x B x decoder_dim.
        last_h, last_c = last_hiddens
        source_context, source_attn_weights = self.source_attention(
            last_h.transpose(0, 1), source_enc, source_mask
        )
        feature_context, _ = self.feature_attention(
            last_h.transpose(0, 1), feature_enc, feature_mask
        )
        # -> B x 1 x 4*hidden_size.
        context = torch.cat([source_context, feature_context], dim=2)
        _, (h, c) = self.decoder(
            torch.cat((embedded, context), 2), (last_h, last_c)
        )
        hidden = h.transpose(0, 1)
        output_probs = self.classifier(torch.cat([hidden, context], dim=2))
        # Ordinary softmax, log will be taken at the end.
        output_probs = nn.functional.softmax(output_probs, dim=2)
        # -> B x 1 x output_size.
        attn_probs = torch.zeros(
            symbol.size(0), self.output_size, device=self.device
        ).unsqueeze(1)
        # Gets the attentions to the source in terms of the output generations.
        # These are the "pointer" distribution.
        # -> B x 1 x output_size.
        attn_probs.scatter_add_(
            2, source_indices.unsqueeze(1), source_attn_weights
        )
        # Probability of generating (from output_probs).
        gen_probs = self.generation_probability(
            context, hidden, embedded
        ).unsqueeze(2)
        gen_scores = gen_probs * output_probs
        ptr_scores = (1 - gen_probs) * attn_probs
        scores = gen_scores + ptr_scores
        # Puts scores in log space.
        scores = torch.log(scores)
        return scores, (h, c)

    def decode(
        self,
        batch_size: int,
        decoder_hiddens: torch.Tensor,
        source_indices: torch.Tensor,
        source_enc: torch.Tensor,
        source_mask: torch.Tensor,
        feature_enc: torch.Tensor,
        feature_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence.

        Args:
            batch_size (int).
            decoder_hiddens (torch.Tensor).
            source_indices (torch.Tensor).
            source_enc (torch.Tensor).
            source_mask (torch.Tensor).
            feature_enc (torch.Tensor).
            feature_mask (torch.Tensor).
            target (torch.Tensor, optional).

        Returns:
            torch.Tensor.
        """
        # Feeds in the first decoder input, as a start tag.
        # -> B x 1
        decoder_input = (
            torch.LongTensor([self.start_idx])
            .to(self.device)
            .repeat(batch_size)
            .unsqueeze(1)
        )
        preds = []
        num_steps = (
            target.size(1) if target is not None else self.max_decode_len
        )
        # Tracks when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size).to(self.device)
        for t in range(num_steps):
            # pred: B x 1 x output_size.
            output, decoder_hiddens = self.decode_step(
                decoder_input,
                decoder_hiddens,
                source_indices,
                source_enc,
                source_mask,
                feature_enc,
                feature_mask,
            )
            preds.append(output.squeeze(1))
            # If we have a target (training) then the next input is the gold
            # symbol for this step (teacher forcing).
            if target is not None:
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise, it must be inference time and we pass the top pred
            # to the next next timestep (student forcing; greedy decoding).
            else:
                decoder_input = self._get_predicted(output)
                # Tracks which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an END symbol.
                if finished.all():
                    break
        preds = torch.stack(preds)
        return preds

    def forward(self, batch: base.Batch) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (base.Batch): tuple of tensors in the batch.

        Returns:
            torch.Tensor.
        """
        # Training mode with targets.
        if len(batch) == 6:
            (
                source,
                source_mask,
                features,
                features_mask,
                target,
                target_mask,
            ) = batch
        # No targets given at inference.
        elif len(batch) == 4:
            source, source_mask, features, features_mask = batch
            target = None
        else:
            raise Error(f"Batch of {len(batch)} elements is invalid")
        batch_size = source.size(0)
        source_encoded, (h_source, c_source) = self.encode(
            source, source_mask, self.encoder
        )
        features_encoded, (h_features, c_features) = self.encode(
            features, features_mask, self.feature_encoder
        )
        h_0 = self.linear_h(torch.cat([h_source, h_features], dim=2))
        c_0 = self.linear_c(torch.cat([c_source, c_features], dim=2))
        if self.beam_width is not None and self.beam_width > 1:
            # preds = self.beam_decode(
            #     batch_size, x_mask, enc_out, beam_width=self.beam_width
            # )
            raise NotImplementedError
        else:
            preds = self.decode(
                batch_size,
                (h_0, c_0),
                source,
                source_encoded,
                source_mask,
                features_encoded,
                features_mask,
                target,
            )
        # -> B x output_size x seq_len.
        preds = preds.transpose(0, 1).transpose(1, 2)
        return preds
