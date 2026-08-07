"""Train GAT or the full architecture on real STEW with subject-disjoint splits.

This command never substitutes synthetic data. Outputs are written to reports/real_stew/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Support both documented module execution and direct script execution from any
# working directory.  When invoked as ``python examples/stew_real_experiment.py``,
# Python otherwise places only ``examples/`` on sys.path and cannot see the
# sibling ``hmb_kuramoto_ode/`` package.
if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from hmb_kuramoto_ode.data.stew import STEWDataset
from hmb_kuramoto_ode.data.preprocessing import RhythmPreprocessor, TrainNormalizer
from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, batch_edges
from hmb_kuramoto_ode.models.attention import EdgeAttention
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.utils.seed import seed_everything

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GATClassifier(nn.Module):
    """Two-layer edge-attention baseline over flattened rhythm nodes."""
    def __init__(self, edge_index: torch.Tensor, features: int = 6, hidden: int = 24):
        super().__init__(); self.register_buffer("edge_index", edge_index)
        self.input=nn.Linear(features,hidden); self.a1=EdgeAttention(hidden); self.a2=EdgeAttention(hidden)
        self.l1=nn.Linear(hidden,hidden); self.l2=nn.Linear(hidden,hidden); self.head=nn.Linear(hidden,2)
    def layer(self,h,attention,projection):
        src,dst=self.edge_index; alpha=attention(h,self.edge_index); out=torch.zeros_like(h)
        out.index_add_(0,dst,alpha[:,None]*projection(h[src])); return torch.relu(out+h)
    def forward(self,x):
        batch,regions,bands,_=x.shape; h=torch.relu(self.input(x.reshape(-1,x.shape[-1])))
        h=self.layer(h,self.a1,self.l1); h=self.layer(h,self.a2,self.l2)
        return self.head(h.reshape(batch,regions*bands,-1).mean(1))

# Cache edge buffers to avoid rebuilding
_edge_cache = {}

def load_windows(root: str, limit_per_record: int):
    """Load windows with progress indication."""
    prep=RhythmPreprocessor(); ds=STEWDataset(root,prep); rows=[]
    records_list = list(ds.records)  # Convert to list for tqdm
    for record in tqdm(records_list, desc="Loading data", unit="record"):
        for start,window in prep.windows(ds.load(record))[:limit_per_record]:
            rows.append((prep.transform_window(window),record.label,record.subject_id,record.path.name,start))
    if len({r[2] for r in rows})<3: raise ValueError("real evaluation needs at least three subjects for train/validation/test")
    return rows

def split_rows(rows):
    subjects=sorted({r[2] for r in rows}); test={subjects[-1]}; validation={subjects[-2]}; train=set(subjects)-test-validation
    # Use list comprehensions more efficiently
    train_rows = [r for r in rows if r[2] in train]
    val_rows = [r for r in rows if r[2] in validation]
    test_rows = [r for r in rows if r[2] in test]
    return [train_rows, val_rows, test_rows], (train,validation,test)

def evaluate(model,x,y,full):
    model.eval()
    with torch.no_grad():
        logits=model(x)["graph_logits"] if full else model(x)
    probability=torch.softmax(logits,1)[:,1].cpu().numpy()
    target=y.cpu().numpy()
    pred=(probability>=.5).astype(int)
    return {
        "accuracy":float(accuracy_score(target,pred)),
        "roc_auc":float(roc_auc_score(target,probability)) if len(set(target))==2 else None,
        "confusion_matrix":confusion_matrix(target,pred,labels=[0,1]).tolist()
    }, probability.tolist()

def get_cached_edge(base, regions, batch_size):
    """Get or create cached edge buffer."""
    key = (batch_size, regions)
    if key not in _edge_cache:
        _edge_cache[key] = batch_edges(base, regions*5, batch_size)
    return _edge_cache[key]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True)
    p.add_argument("--model",choices=("gat","full"),default="full")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--windows-per-record",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=16, help="Batch size for training")
    a=p.parse_args()
    seed_everything(7)
    
    # Load data with progress
    rows=load_windows(a.data_root,a.windows_per_record)
    (train,val,test),subjects=split_rows(rows)
    
    # Normalize and create tensors
    normalizer=TrainNormalizer().fit(np.stack([r[0] for r in train]),[r[2] for r in train])
    def tensors(part): 
        features = normalizer.transform(np.stack([r[0] for r in part]))
        return torch.tensor(features, dtype=torch.float32).to(device), torch.tensor([r[1] for r in part], dtype=torch.long).to(device)
    
    xtr,ytr=tensors(train)
    xv,yv=tensors(val)
    xt,yt=tensors(test)
    regions=xtr.shape[1]
    
    # Build graph and model
    base,_=build_hierarchical_graph(regions)
    full=a.model=="full"
    
    # Pre-compute edge buffers for different batch sizes
    train_batch_size = min(a.batch_size, len(train))
    val_batch_size = len(val)
    test_batch_size = len(test)
    
    if full:
        model=HierarchicalKuramotoODE(get_cached_edge(base, regions, train_batch_size), hidden=16).to(device)
    else:
        model=GATClassifier(get_cached_edge(base, regions, train_batch_size)).to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=.01)
    history=[]
    best=None
    
    # Training loop with tqdm
    for epoch in tqdm(range(a.epochs), desc="Training", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        
        # Use cached edges for training
        if full:
            model.field.edge_index = get_cached_edge(base, regions, train_batch_size)
        else:
            model.edge_index = get_cached_edge(base, regions, train_batch_size)
        
        logits=model(xtr)["graph_logits"] if full else model(xtr)
        loss=nn.functional.cross_entropy(logits,ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        
        # Validation
        if full:
            model.field.edge_index = get_cached_edge(base, regions, val_batch_size)
        else:
            model.edge_index = get_cached_edge(base, regions, val_batch_size)
        
        with torch.no_grad():
            val_logits=model(xv)["graph_logits"] if full else model(xv)
            val_loss=float(nn.functional.cross_entropy(val_logits,yv))
        
        val_metrics,_=evaluate(model,xv,yv,full)
        history.append({"epoch":epoch+1,"train_loss":float(loss),"val_loss":val_loss,"val_accuracy":val_metrics["accuracy"]})
        
        # Save best model
        state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.endswith("edge_index")}
        if best is None or val_metrics["accuracy"]>best[0]:
            best=(val_metrics["accuracy"],state)
        
        # Restore training edges
        if full:
            model.field.edge_index = get_cached_edge(base, regions, train_batch_size)
        else:
            model.edge_index = get_cached_edge(base, regions, train_batch_size)
    # Final evaluation on test set
    model.load_state_dict(best[1],strict=False)
    if full:
        model.field.edge_index = get_cached_edge(base, regions, test_batch_size)
    else:
        model.edge_index = get_cached_edge(base, regions, test_batch_size)
    
    metrics,predictions=evaluate(model,xt,yt,full)
    
    # Save results
    out=Path("reports/real_stew")
    out.mkdir(parents=True,exist_ok=True)
    
    payload={
        "dataset":"STEW real files",
        "model":a.model,
        "seed":7,
        "subjects":{
            "train":sorted(subjects[0]),
            "validation":sorted(subjects[1]),
            "test":sorted(subjects[2])
        },
        "metrics":metrics,
        "history":history,
        "test_predictions":predictions
    }
    
    (out/f"{a.model}_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")
    
    # Plot training curves
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    epochs=[h["epoch"] for h in history]
    ax.plot(epochs,[h["train_loss"] for h in history],label="train loss")
    ax.plot(epochs,[h["val_loss"] for h in history],label="validation loss")
    ax2=ax.twinx()
    ax2.plot(epochs,[h["val_accuracy"] for h in history],color="orange",linestyle="--",label="validation accuracy")
    ax.set(xlabel="epoch",ylabel="cross-entropy loss")
    ax2.set_ylabel("validation accuracy")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out/f"{a.model}_loss_validation.png",dpi=180)
    fig.savefig(out/f"{a.model}_loss_validation.svg")
    plt.close(fig)
    
    # Write report
    (out/f"{a.model}_report.md").write_text(
        f"# Real STEW {a.model.upper()} report\n\n"
        f"Subject-disjoint split: `{payload['subjects']}`.\n\n"
        f"- ROC-AUC: `{metrics['roc_auc']}`\n"
        f"- Accuracy: `{metrics['accuracy']}`\n"
        f"- Confusion matrix (rows=true, columns=predicted): `{metrics['confusion_matrix']}`\n\n"
        f"![Training loss and validation accuracy]({a.model}_loss_validation.svg)\n"
    )
    
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
