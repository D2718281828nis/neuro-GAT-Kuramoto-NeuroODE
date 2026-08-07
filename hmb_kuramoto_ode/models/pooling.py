import torch
from torch import nn

class HierarchicalAttentionPooling(nn.Module):
    def __init__(self,hidden): super().__init__(); self.rhythm=nn.Linear(hidden,1); self.region=nn.Linear(hidden,1)
    def forward(self,z,channel_mask=None):
        # z [B,R,5,H]
        a=torch.softmax(self.rhythm(z).squeeze(-1),-1); regions=(a[...,None]*z).sum(2)
        logits=self.region(regions).squeeze(-1)
        if channel_mask is not None: logits=logits.masked_fill(~channel_mask,-torch.inf)
        q=torch.softmax(logits,-1); graph=(q[...,None]*regions).sum(1)
        return graph,regions,{"rhythm":a,"region":q}

