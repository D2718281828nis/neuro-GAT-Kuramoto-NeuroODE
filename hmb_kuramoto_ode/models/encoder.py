import torch
from torch import nn

class BiomedicalEncoder(nn.Module):
    """Band-specific feature projection; latent dim 0 is the angular state."""
    def __init__(self, features=6, hidden=16, max_regions=64, dropout=.1):
        super().__init__(); self.band=nn.ModuleList([nn.Linear(features,hidden) for _ in range(5)])
        self.region=nn.Embedding(max_regions,hidden); self.norm=nn.LayerNorm(features)
        self.residual=nn.Linear(features,hidden); self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        b,r,nb,f=x.shape; x=self.norm(x); region=self.region(torch.arange(r,device=x.device))[None,:,None]
        z=torch.stack([self.band[k](x[:,:,k]) for k in range(nb)],2)+self.residual(x)+region
        # Phase is explicitly initialized from atan2(sin phase feature, cos phase feature).
        z=z.clone(); z[...,0]=torch.atan2(x[...,3],x[...,4]); return self.dropout(torch.tanh(z))

