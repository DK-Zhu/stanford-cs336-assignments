import torch.nn as nn
import torch
from math import sqrt
from einops import einsum
from jaxtyping import Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        weight = torch.empty(out_features, in_features)
        sigma = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            weight,
            mean = 0,
            std = sigma,
            a = -3 * sigma,
            b = 3 * sigma
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        embedding_mat = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(
            embedding_mat,
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
        self.embedding_matrix = nn.Parameter(embedding_mat)

    def forward(self, token_ids: Int[Tensor, "batch seq_len"]) -> Int[Tensor, "batch seq_len d_model"]:
        embeddings = self.embedding_matrix[token_ids]
        return embeddings