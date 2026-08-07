import torch
from torch import nn
class LinkDecoder(nn.Module):
    def __init__(self,hidden): super().__init__(); self.mlp=nn.Sequential(nn.Linear(hidden*4,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,z,edges):
        u,v=edges; return self.mlp(torch.cat([z[u],z[v],abs(z[u]-z[v]),z[u]*z[v]],-1)).squeeze(-1)

