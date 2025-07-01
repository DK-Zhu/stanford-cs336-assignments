import torch.nn as nn
import torch
from math import sqrt
from einops import einsum, reduce, rearrange
from jaxtyping import Int, Float
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
    
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.eps = eps
        self.device = device
        self.dtype = dtype

        g = torch.ones(d_model)
        self.g = nn.Parameter(g)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = (reduce(x.pow(2), "batch seq_len d_model -> batch seq_len", "mean") + self.eps).sqrt()
        result = x / rearrange(RMS, "batch seq_len -> batch seq_len 1") * rearrange(self.g, "d_model -> 1 1 d_model")
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        # Compute d_ff as 8/3 * d_model, rounded up to nearest multiple of 64
        d_ff = int(((8 * d_model / 3) // 64 + 1) * 64) if d_ff is None else d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
