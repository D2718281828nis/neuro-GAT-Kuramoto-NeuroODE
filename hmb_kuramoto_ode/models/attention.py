"""Dependency-free edge attention normalized over incoming neighbors."""
import torch
from torch import nn

def segment_softmax(scores: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    out=torch.zeros_like(scores)
    for i in range(n):
        mask=dst==i
        if mask.any(): out[mask]=torch.softmax(scores[mask], dim=0)
    return out

class EdgeAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.score=nn.Sequential(nn.Linear(hidden*2+1,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,h,edge_index,edge_weight=None):
        src,dst=edge_index; w=torch.ones(src.numel(),device=h.device) if edge_weight is None else edge_weight
        score=self.score(torch.cat([h[src],h[dst],w[:,None]],-1)).squeeze(-1)
        return segment_softmax(score,dst,h.shape[0])

