"""Link prediction: reconstruct the hierarchical electrode/rhythm graph from
window-conditioned oscillator embeddings.

Positive edges are the canonical same-band spatial and local cross-frequency
links from data/graph_builder.py; negatives are canonical non-edges within the
same window. The identity of each pair (edge or not) never changes across
windows -- what varies is whether the model, given only the real EEG-driven
node embeddings z(t1) for that window and subject, can still recover which
pairs are connected. Evaluated on real, subject-disjoint STEW windows. See
TASK.md in this folder for the write-up.

Usage:
    python tasks/link_prediction/evaluate.py --data-root dataset
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
from hmb_kuramoto_ode.training.metrics import link_metrics
from hmb_kuramoto_ode.utils.seed import seed_everything
from tasks.common import (
    fit_and_transform,
    hierarchical_edges,
    limit_subjects,
    load_stew_windows,
    plot_loss_curve,
    plot_roc_pr,
    sample_link_pairs,
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
    x_train, _ = tensors["train"]
    x_val, _ = tensors["validation"]
    x_test, _ = tensors["test"]
    regions = x_train.shape[1]

    base_edges, _, _ = hierarchical_edges(regions, 1)
    model = HierarchicalKuramotoODE(base_edges, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def prepared(x, seed):
        batch = x.shape[0]
        edges, base, _ = hierarchical_edges(regions, batch)
        pairs, labels = sample_link_pairs(base, regions, batch, seed=seed)
        return edges, pairs, labels

    train_edges, train_pairs, train_labels = prepared(x_train, args.seed + 1)
    val_edges, val_pairs, val_labels = prepared(x_val, args.seed + 2)
    test_edges, test_pairs, test_labels = prepared(x_test, args.seed + 3)

    history = []
    best_state, best_val_auroc = None, -1.0
    for epoch in tqdm(range(1, args.epochs + 1), desc="link_prediction", unit="epoch"):
        model.train()
        model.field.edge_index = train_edges
        optimizer.zero_grad()
        logits = model(x_train, edge_pairs=train_pairs)["link_logits"]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            model.field.edge_index = val_edges
            val_logits = model(x_val, edge_pairs=val_pairs)["link_logits"]
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, val_labels))
            val_score = link_metrics(val_labels.numpy(), val_logits.numpy())
        history.append({"epoch": epoch, "train_loss": float(loss), "val_loss": val_loss,
                         "val_auroc": val_score["auroc"]})
        if val_score["auroc"] > best_val_auroc:
            best_val_auroc = val_score["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("edge_index")}

    model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        model.field.edge_index = test_edges
        test_logits = model(x_test, edge_pairs=test_pairs)["link_logits"]
    test_score = link_metrics(test_labels.numpy(), test_logits.numpy())
    probability = torch.sigmoid(test_logits).numpy()
    prediction = (probability >= 0.5).astype(int)
    accuracy = float((prediction == test_labels.numpy()).mean())
    confusion = [[0, 0], [0, 0]]
    for target, pred in zip(test_labels.numpy().astype(int), prediction):
        confusion[target][pred] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, OUT / "loss_curve", secondary_key="val_auroc", secondary_label="validation ROC-AUC")
    plot_roc_pr(test_labels.numpy(), probability, OUT / "roc_pr", title="Link prediction")
    save_confusion_matrix(confusion, OUT / "confusion_matrix")

    metrics = {
        "edges": int(test_labels.numel()),
        "auroc": test_score["auroc"],
        "average_precision": test_score["average_precision"],
        "accuracy_at_0.5": accuracy,
        "confusion_matrix": confusion,
    }
    payload = {
        "task": "link_prediction",
        "dataset": "real STEW recordings under " + str(Path(args.data_root).resolve()),
        "subjects": subject_ids,
        "windows": {fold: len(rows_) for fold, rows_ in split.items()},
        "args": vars(args),
        "history": history,
        "test_metrics": metrics,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "results.md").write_text(f"""# Link prediction — measured results

> Real STEW recordings from `{args.data_root}`, subject-disjoint split. Not a synthetic fixture.
> Edge identity (positive/negative) is fixed by the hierarchical topology; only the
> node embeddings that decode it come from real, window-specific EEG dynamics.

Subjects: train={subject_ids['train']}, validation={subject_ids['validation']}, test={subject_ids['test']}
Windows: train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}

| Metric | Test value |
|---|---:|
| Edges evaluated | {metrics['edges']} |
| ROC-AUC | {metrics['auroc']:.4f} |
| Average precision | {metrics['average_precision']:.4f} |
| Accuracy at 0.5 | {metrics['accuracy_at_0.5']:.4f} |

Confusion matrix (rows=true [no-edge, edge], columns=predicted [no-edge, edge]):

```text
{confusion[0]}
{confusion[1]}
```

![Loss and validation ROC-AUC](loss_curve.svg)
![ROC and PR curves](roc_pr.svg)
![Confusion matrix](confusion_matrix.svg)

## Reproduce

```bash
python tasks/link_prediction/evaluate.py --data-root {args.data_root}
```
""")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
