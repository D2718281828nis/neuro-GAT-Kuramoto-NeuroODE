"""Train GAT or the full architecture on real STEW with subject-disjoint splits.

This command never substitutes synthetic data. Outputs are written to reports/real_stew/.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
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

def load_windows(root: str, limit_per_record: int):
    prep=RhythmPreprocessor(); ds=STEWDataset(root,prep); rows=[]
    for record in ds.records:
        for start,window in prep.windows(ds.load(record))[:limit_per_record]:
            rows.append((prep.transform_window(window),record.label,record.subject_id,record.path.name,start))
    if len({r[2] for r in rows})<3: raise ValueError("real evaluation needs at least three subjects for train/validation/test")
    return rows

def split_rows(rows):
    subjects=sorted({r[2] for r in rows}); test={subjects[-1]}; validation={subjects[-2]}; train=set(subjects)-test-validation
    return [[r for r in rows if r[2] in group] for group in (train,validation,test)], (train,validation,test)

def evaluate(model,x,y,full):
    model.eval()
    with torch.no_grad(): logits=model(x)["graph_logits"] if full else model(x)
    probability=torch.softmax(logits,1)[:,1].cpu().numpy(); target=y.cpu().numpy(); pred=(probability>=.5).astype(int)
    return {"accuracy":float(accuracy_score(target,pred)),"roc_auc":float(roc_auc_score(target,probability)) if len(set(target))==2 else None,"confusion_matrix":confusion_matrix(target,pred,labels=[0,1]).tolist()}, probability.tolist()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--model",choices=("gat","full"),default="full"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--windows-per-record",type=int,default=4); a=p.parse_args(); seed_everything(7)
    rows=load_windows(a.data_root,a.windows_per_record); (train,val,test),subjects=split_rows(rows)
    normalizer=TrainNormalizer().fit(np.stack([r[0] for r in train]),[r[2] for r in train])
    def tensors(part): return torch.tensor(normalizer.transform(np.stack([r[0] for r in part]))),torch.tensor([r[1] for r in part])
    xtr,ytr=tensors(train); xv,yv=tensors(val); xt,yt=tensors(test); regions=xtr.shape[1]
    base,_=build_hierarchical_graph(regions); full=a.model=="full"
    if full: model=HierarchicalKuramotoODE(batch_edges(base,regions*5,len(train)),hidden=16)
    else: model=GATClassifier(batch_edges(base,regions*5,len(train)))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01); history=[]; best=None
    for epoch in range(a.epochs):
        model.train(); optimizer.zero_grad(); logits=model(xtr)["graph_logits"] if full else model(xtr); loss=nn.functional.cross_entropy(logits,ytr); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        # Rebind graph-sized edge buffers before validation for disconnected batches.
        if full: model.field.edge_index=batch_edges(base,regions*5,len(val))
        else: model.edge_index=batch_edges(base,regions*5,len(val))
        with torch.no_grad():
            val_logits=model(xv)["graph_logits"] if full else model(xv)
            val_loss=float(nn.functional.cross_entropy(val_logits,yv))
        val_metrics,_=evaluate(model,xv,yv,full); history.append({"epoch":epoch+1,"train_loss":float(loss),"val_loss":val_loss,"val_accuracy":val_metrics["accuracy"]})
        state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.endswith("edge_index")}; best=(val_metrics["accuracy"],state) if best is None or val_metrics["accuracy"]>best[0] else best
        if full: model.field.edge_index=batch_edges(base,regions*5,len(train))
        else: model.edge_index=batch_edges(base,regions*5,len(train))
    model.load_state_dict(best[1],strict=False)
    if full: model.field.edge_index=batch_edges(base,regions*5,len(test))
    else: model.edge_index=batch_edges(base,regions*5,len(test))
    metrics,predictions=evaluate(model,xt,yt,full); out=Path("reports/real_stew"); out.mkdir(parents=True,exist_ok=True)
    payload={"dataset":"STEW real files","model":a.model,"seed":7,"subjects":{"train":sorted(subjects[0]),"validation":sorted(subjects[1]),"test":sorted(subjects[2])},"metrics":metrics,"history":history,"test_predictions":predictions}
    (out/f"{a.model}_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); epochs=[h["epoch"] for h in history]; ax.plot(epochs,[h["train_loss"] for h in history],label="train loss"); ax.plot(epochs,[h["val_loss"] for h in history],label="validation loss"); ax2=ax.twinx(); ax2.plot(epochs,[h["val_accuracy"] for h in history],color="orange",linestyle="--",label="validation accuracy"); ax.set(xlabel="epoch",ylabel="cross-entropy loss"); ax2.set_ylabel("validation accuracy"); ax.legend(loc="upper left"); ax2.legend(loc="upper right"); fig.tight_layout(); fig.savefig(out/f"{a.model}_loss_validation.png",dpi=180); fig.savefig(out/f"{a.model}_loss_validation.svg"); plt.close(fig)
    (out/f"{a.model}_report.md").write_text(f"# Real STEW {a.model.upper()} report\n\nSubject-disjoint split: `{payload['subjects']}`.\n\n- ROC-AUC: `{metrics['roc_auc']}`\n- Accuracy: `{metrics['accuracy']}`\n- Confusion matrix (rows=true, columns=predicted): `{metrics['confusion_matrix']}`\n\n![Training loss and validation accuracy]({a.model}_loss_validation.svg)\n")
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
