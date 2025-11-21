from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.typing import Adj

from .prototypes import STGNN


class SimpleSTGNN(STGNN):
    """
    A minimal STGNN that applies a single diffusion step per time slice using a
    fixed graph (edge_index, edge_weight). It uses the base class encoder/decoder.
    """

    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.proj = nn.Identity()

    def _build_sparse_adj(self, edge_index: Tensor, edge_weight: Optional[Tensor]) -> torch.Tensor:
        """
        Create a row-normalized sparse adjacency matrix A in torch.sparse_coo format.
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
        # Row-normalize by out-degree
        n = int(edge_index.max().item()) + 1
        deg = torch.zeros(n, device=edge_index.device, dtype=torch.float32)
        deg = deg.index_add(0, edge_index[0], edge_weight)
        deg = torch.clamp(deg, min=1e-6)
        norm_w = edge_weight / deg[edge_index[0]]
        A = torch.sparse_coo_tensor(edge_index, norm_w, (n, n))
        A = A.coalesce()
        return A

    def stmp(self, x: Tensor, edge_index: Adj,
             edge_weight: Optional[Tensor] = None,
             emb: Optional[Tensor] = None) -> Tensor:
        # x: [B, T, N, H]
        B, T, N, H = x.shape
        A = self._build_sparse_adj(edge_index, edge_weight)
        out = torch.empty_like(x)
        # For each (b, t), apply: X' = alpha * X + (1-alpha) * A @ X
        # where X: [N, H]
        for b in range(B):
            for t in range(T):
                X = x[b, t]  # [N, H]
                AX = torch.sparse.mm(A, X)
                out[b, t] = self.alpha * X + (1.0 - self.alpha) * AX
        return out


