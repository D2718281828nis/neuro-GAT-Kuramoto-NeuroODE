import torch
def sample_negative_edges(nodes,positive,count,generator=None):
    known={tuple(sorted(e)) for e in positive.T.tolist()}; available=[(i,j) for i in range(nodes) for j in range(i+1,nodes) if (i,j) not in known]
    if count>len(available): raise ValueError("not enough unique negative edges")
    order=torch.randperm(len(available),generator=generator)[:count]; return torch.tensor([available[i] for i in order.tolist()],dtype=torch.long).T
