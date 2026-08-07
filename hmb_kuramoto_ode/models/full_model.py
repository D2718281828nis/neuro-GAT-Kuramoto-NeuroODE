import torch
from torch import nn
from .encoder import BiomedicalEncoder
from .ode_func import KuramotoGrandVectorField
from .ode_solver import integrate
from .pooling import HierarchicalAttentionPooling
from .heads import LinkDecoder

class HierarchicalKuramotoODE(nn.Module):
    def __init__(self,edge_index,features=6,hidden=16,classes=2,t1=.2,step_size=.1,**ablations):
        super().__init__(); self.encoder=BiomedicalEncoder(features,hidden); self.field=KuramotoGrandVectorField(hidden,edge_index,**ablations)
        self.pool=HierarchicalAttentionPooling(hidden); self.graph_head=nn.Linear(hidden,classes); self.node_head=nn.Linear(hidden,features); self.link_head=LinkDecoder(hidden)
        self.t1=t1; self.step_size=step_size
    def forward(self,x,edge_pairs=None,channel_mask=None,return_trajectory=False):
        b,r,nb,_=x.shape; z0=self.encoder(x).reshape(b*r*nb,-1); self.field.set_static_attention(z0); self.field.nfe=0
        result=integrate(self.field,z0,t1=self.t1,step_size=self.step_size,return_trajectory=return_trajectory)
        z,trajectory=result if return_trajectory else (result,None); shaped=z.reshape(b,r,nb,-1); graph,regions,attn=self.pool(shaped,channel_mask)
        out={"graph_logits":self.graph_head(graph),"node_prediction":self.node_head(shaped),"embedding":z,"attention":attn,"nfe":self.field.nfe}
        if edge_pairs is not None: out["link_logits"]=self.link_head(z,edge_pairs)
        if trajectory is not None: out["trajectory"]=trajectory
        return out

