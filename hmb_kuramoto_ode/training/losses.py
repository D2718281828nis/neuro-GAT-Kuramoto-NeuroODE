import torch
from torch.nn import functional as F
def multitask_loss(out,graph_target,node_target,link_target=None,weights=(1.,1.,1.)):
    terms={"graph":F.cross_entropy(out["graph_logits"],graph_target),"node":F.mse_loss(out["node_prediction"],node_target)}
    terms["edge"]=F.binary_cross_entropy_with_logits(out["link_logits"],link_target) if link_target is not None else torch.zeros((),device=graph_target.device)
    return sum(w*terms[k] for w,k in zip(weights,("graph","node","edge"))),terms
