"""Explicit tensor contracts shared by data and models."""
from dataclasses import dataclass
import torch

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DEFAULT_CHANNELS = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")

@dataclass
class GraphBatch:
    """Rhythm graph: x [B,R,5,F], edges [2,E], optional masks/weights."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor | None = None
    channel_mask: torch.Tensor | None = None

    def validate(self) -> None:
        if self.x.ndim != 4 or self.x.shape[2] != len(BANDS):
            raise ValueError(f"x must be [batch, regions, 5, features], got {tuple(self.x.shape)}")
        if not torch.isfinite(self.x).all(): raise ValueError("x contains NaN/Inf")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2: raise ValueError("edge_index must be [2,E]")
        n = self.x.shape[0] * self.x.shape[1] * self.x.shape[2]
        if self.edge_index.numel() and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index contains out-of-range node indices")
        if self.edge_type.shape != (self.edge_index.shape[1],): raise ValueError("edge_type length mismatch")

