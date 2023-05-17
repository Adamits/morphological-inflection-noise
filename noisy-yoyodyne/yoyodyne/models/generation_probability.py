"""Generation probability."""

import math

import torch
from torch import nn


class GenerationProbability(nn.Module):
    """Calculates the generation probability for a pointer generator."""

    stdev = 1 / math.sqrt(100)

    W_attn: nn.Linear
    W_hs: nn.Linear
    W_inp: nn.Linear
    bias: nn.Parameter

    def __init__(self, embedding_size: int, hidden_size: int, attn_size: int):
        """Initializes the generation probability operator.

        Args:
            embedding_size (int): embedding dimensions.
            hidden_size (int): decoder hidden state dimensions.
            attn_size (int): dimensions of combined encoder attentions.
        """
        super().__init__()
        self.W_attn = nn.Linear(attn_size, 1, bias=False)
        self.W_hs = nn.Linear(hidden_size, 1, bias=False)
        self.W_inp = nn.Linear(embedding_size, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.uniform_(-self.stdev, self.stdev)

    def forward(
        self, h_attn: torch.Tensor, dec_hs: torch.Tensor, inp: torch.Tensor
    ) -> torch.Tensor:
        """Computes Wh * ATTN_t + Ws * HIDDEN_t + Wy * Y_{t-1} + b.

        Args:
            h_attn (torch.Tensor): combined context vector over source and
                features of shape B x 1 x attn_size.
            dec_hs (torch.Tensor): decoder hidden state of shape
                B x 1 x hidden_size.
            inp (torch.Tensor): decoder input of shape B x 1 x embedding_size.

        Returns:
            (torch.Tensor): generation probability of shape B.
        """
        # -> B x 1 x 1.
        p_gen = self.W_attn(h_attn) + self.W_hs(dec_hs)
        p_gen += self.W_inp(inp) + self.bias.expand(h_attn.size(0), 1, -1)
        # -> B.
        p_gen = torch.sigmoid(p_gen.squeeze(1))
        return p_gen
