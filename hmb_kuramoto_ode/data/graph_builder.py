"""Typed hierarchical electrode/rhythm graph construction."""
import torch
SPATIAL_SAME_BAND, CROSS_FREQUENCY_LOCAL, CROSS_BAND_SPATIAL, GLOBAL_OR_LATENT = range(4)

def build_hierarchical_graph(regions: int, spatial_edges: list[tuple[int,int]] | None=None, cross_band_spatial: bool=False):
    spatial_edges = spatial_edges or [(i, i+1) for i in range(regions-1)]
    edges=[]; types=[]
    def add(u,v,t): edges.extend(((u,v),(v,u))); types.extend((t,t))
    for i,j in spatial_edges:
        for b in range(5): add(i*5+b,j*5+b,SPATIAL_SAME_BAND)
        if cross_band_spatial:
            for b in range(4): add(i*5+b,j*5+b+1,CROSS_BAND_SPATIAL)
    for i in range(regions):
        for b in range(5):
            for c in range(b+1,5): add(i*5+b,i*5+c,CROSS_FREQUENCY_LOCAL)
    return torch.tensor(edges,dtype=torch.long).T.contiguous(), torch.tensor(types,dtype=torch.long)

def batch_edges(edge_index: torch.Tensor, nodes_per_graph: int, batch_size: int) -> torch.Tensor:
    return torch.cat([edge_index + k*nodes_per_graph for k in range(batch_size)], dim=1)
