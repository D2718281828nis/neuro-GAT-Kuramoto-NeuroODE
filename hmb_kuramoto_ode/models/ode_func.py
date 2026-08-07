"""Kuramoto-attention-GRAND continuous-time vector field."""
import torch
from torch import nn
from .attention import EdgeAttention

class KuramotoGrandVectorField(nn.Module):
    def __init__(self,hidden,edge_index,dynamic_attention=True,cross_frequency=True,grand=True,kuramoto=True,residual=True):
        super().__init__(); self.hidden=hidden; self.register_buffer("edge_index",edge_index); self.attention=EdgeAttention(hidden)
        self.frequency=nn.Linear(hidden,1); self.raw_coupling=nn.Parameter(torch.zeros(4)); self.gate=nn.Linear(hidden,hidden)
        self.residual=nn.Sequential(nn.Linear(hidden+1,hidden),nn.Tanh(),nn.Linear(hidden,hidden))
        self.dynamic_attention=dynamic_attention; self.cross_frequency=cross_frequency; self.use_grand=grand; self.use_kuramoto=kuramoto; self.use_residual=residual
        self.static_alpha=None; self.nfe=0; self.last_gate=None
    @property
    def coupling(self): return 2*torch.sigmoid(self.raw_coupling)
    def set_static_attention(self,h): self.static_alpha=self.attention(h,self.edge_index).detach() if not self.dynamic_attention else None
    def forward(self,t,h):
        self.nfe+=1; src,dst=self.edge_index; alpha=self.attention(h,self.edge_index) if self.dynamic_attention or self.static_alpha is None else self.static_alpha
        phase=h[:,0]; omega=torch.pi*torch.tanh(self.frequency(h).squeeze(-1)); kur=torch.zeros_like(h); grand=torch.zeros_like(h)
        kur[:,0].index_add_(0,dst,alpha*self.coupling[0]*torch.sin(phase[src]-phase[dst]))
        grand.index_add_(0,dst,alpha[:,None]*(h[src]-h[dst]))
        if self.cross_frequency:
            same_region=(src//5)==(dst//5); different=(src%5)!=(dst%5); m=same_region&different
            kur[:,0].index_add_(0,dst[m],alpha[m]*self.coupling[1]*torch.sin(phase[src[m]]-phase[dst[m]]))
        kur[:,0]+=omega; g=torch.sigmoid(self.gate(h)); self.last_gate=g
        out=g*kur*(1 if self.use_kuramoto else 0)+(1-g)*grand*(1 if self.use_grand else 0)
        if self.use_residual: out=out+self.residual(torch.cat([h,t.reshape(1,1).expand(h.shape[0],1)],-1))*.1
        return out
