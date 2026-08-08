"""Graph prediction: classify a STEW window as high- or low-workload.

Trains the full HierarchicalKuramotoODE on real, subject-disjoint STEW windows
and evaluates hmb_kuramoto_ode.training.metrics.classification_metrics on a
held-out set of subjects. See TASK.md in this folder for the write-up.

Usage:
    python tasks/graph_prediction/evaluate.py --data-root dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    _repository_root = Path(__file__).resolve().parents[2]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))

import torch
from torch import nn
from tqdm import tqdm

from hmb_kuramoto_ode.analysis.visualization import save_confusion_matrix
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.training.metrics import classification_metrics
from hmb_kuramoto_ode.utils.seed import seed_everything
from tasks.common import (
    fit_and_transform,
    hierarchical_edges,
    limit_subjects,
    load_stew_windows,
    plot_loss_curve,
    plot_roc_pr,
    subject_disjoint_split,
)

OUT = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--max-subjects", type=int, default=12)
    parser.add_argument("--windows-per-record", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed)

    rows = limit_subjects(load_stew_windows(args.data_root, args.windows_per_record), args.max_subjects, args.seed)
    split, subject_ids = subject_disjoint_split(rows, seed=args.seed)
    tensors, _ = fit_and_transform(split)
    x_train, y_train = tensors["train"]
    x_val, y_val = tensors["validation"]
    x_test, y_test = tensors["test"]
    regions = x_train.shape[1]

    base_edges, _, _ = hierarchical_edges(regions, 1)
    model = HierarchicalKuramotoODE(base_edges, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def set_edges(batch_size):
        edges, _, _ = hierarchical_edges(regions, batch_size)
        model.field.edge_index = edges

    history = []
    best_state, best_val_acc = None, -1.0
    for epoch in tqdm(range(1, args.epochs + 1), desc="graph_prediction", unit="epoch"):
        model.train()
        set_edges(x_train.shape[0])
        optimizer.zero_grad()
        logits = model(x_train)["graph_logits"]
        loss = nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            set_edges(x_val.shape[0])
            val_logits = model(x_val)["graph_logits"]
            val_loss = float(nn.functional.cross_entropy(val_logits, y_val))
            val_prob = torch.softmax(val_logits, 1)
            val_metrics = classification_metrics(y_val.numpy(), val_prob.numpy())
        history.append({"epoch": epoch, "train_loss": float(loss), "val_loss": val_loss,
                         "val_accuracy": val_metrics["accuracy"]})
        if val_metrics["balanced_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["balanced_accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("edge_index")}

    model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        set_edges(x_test.shape[0])
        test_logits = model(x_test)["graph_logits"]
        test_prob = torch.softmax(test_logits, 1).numpy()
    metrics = classification_metrics(y_test.numpy(), test_prob)

    OUT.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, OUT / "loss_curve", secondary_key="val_accuracy", secondary_label="validation accuracy")
    plot_roc_pr(y_test.numpy(), test_prob[:, 1], OUT / "roc_pr", title="Graph prediction")
    save_confusion_matrix(metrics["confusion_matrix"], OUT / "confusion_matrix")

    payload = {
        "task": "graph_prediction",
        "dataset": "real STEW recordings under " + str(Path(args.data_root).resolve()),
        "subjects": subject_ids,
        "windows": {fold: len(rows_) for fold, rows_ in split.items()},
        "args": vars(args),
        "history": history,
        "test_metrics": metrics,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "results.md").write_text(f"""# Graph prediction — measured results

> Real STEW recordings from `{args.data_root}`, subject-disjoint split. Not a synthetic fixture.

Subjects: train={subject_ids['train']}, validation={subject_ids['validation']}, test={subject_ids['test']}
Windows: train={split_count(split, 'train')}, validation={split_count(split, 'validation')}, test={split_count(split, 'test')}

| Metric | Test value |
|---|---:|
| Accuracy | {metrics['accuracy']:.4f} |
| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| ROC-AUC | {metrics.get('auroc', float('nan')):.4f} |
| AUPRC | {metrics.get('auprc', float('nan')):.4f} |
| ECE | {metrics['ece']:.4f} |

Confusion matrix (rows=true [low, high], columns=predicted [low, high]):

```text
{metrics['confusion_matrix'][0]}
{metrics['confusion_matrix'][1]}
```

![Loss and validation accuracy](loss_curve.svg)
![ROC and PR curves](roc_pr.svg)
![Confusion matrix](confusion_matrix.svg)

## Reproduce

```bash
python tasks/graph_prediction/evaluate.py --data-root {args.data_root}
```
""")
    print(json.dumps(metrics, indent=2))


def split_count(split, fold):
    return len(split[fold])


if __name__ == "__main__":
    main()
