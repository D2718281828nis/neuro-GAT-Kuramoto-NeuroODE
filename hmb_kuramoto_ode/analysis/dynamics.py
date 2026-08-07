import torch
def dynamics_metrics(z,edge_index):
    src,dst=edge_index; phase=z[:,0]; return {"dirichlet_energy":float(((z[src]-z[dst])**2).sum(-1).mean()),"feature_variance":float(z.var()),"phase_order":float(torch.abs(torch.exp(1j*phase).mean()))}
