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
from hmb_kuramoto_ode.contracts import DEFAULT_CHANNELS
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.training.metrics import classification_metrics
from hmb_kuramoto_ode.utils.seed import seed_everything
from tasks.common import (
    ELECTRODE_POSITIONS,
    band_legend_handles,
    draw_hierarchical_topology,
    fit_and_transform,
    hierarchical_edges,
    limit_subjects,
    load_stew_windows,
    plot_loss_curve,
    plot_roc_pr,
    save_topology_figure,
    subject_disjoint_split,
)

OUT = Path(__file__).parent / "results"
CLASS_NAMES = ["low", "high"]


def plot_pooling_topology(regions, rhythm_attention, region_attention, true_label, predicted_label, path):
    """Draw the hierarchical graph for one real held-out window, sized by the
    model's own learned pooling attention: band-node size = rhythm attention
    within its electrode, hub-edge width = electrode's weight in the final
    graph embedding that graph_head classifies. This is the literal pooling
    computation in models/pooling.py::HierarchicalAttentionPooling, not a
    schematic -- the weights come from a real forward pass.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    node_size = 30 + 600 * rhythm_attention.reshape(-1)
    draw_hierarchical_topology(ax, regions, node_size=node_size)
    ax.scatter([0], [0], s=280, color="#111827", zorder=5, marker="s")
    ax.text(0, 0, "graph\nembedding", color="white", fontsize=6.5, ha="center", va="center",
            zorder=6, fontweight="bold")
    for electrode, name in enumerate(DEFAULT_CHANNELS[:regions]):
        cx, cy = ELECTRODE_POSITIONS[name]
        weight = float(region_attention[electrode])
        ax.plot([cx, 0], [cy, 0], color="#f97316", linewidth=0.5 + 14 * weight, alpha=0.3 + 0.6 * weight, zorder=1)
    handles = band_legend_handles() + [plt.Line2D([0], [0], color="#f97316", linewidth=3,
                                                    label="region -> graph attention")]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    condition = {0: "low", 1: "high"}
    save_topology_figure(fig, path,
                          f"Pooling attention, real test window (true={condition[true_label]}, "
                          f"predicted={condition[predicted_label]})")


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
        test_out = model(x_test)
        test_logits = test_out["graph_logits"]
        test_prob = torch.softmax(test_logits, 1).numpy()
    metrics = classification_metrics(y_test.numpy(), test_prob)

    OUT.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, OUT / "loss_curve", secondary_key="val_accuracy", secondary_label="validation accuracy")
    plot_roc_pr(y_test.numpy(), test_prob[:, 1], OUT / "roc_pr", title="Graph prediction")
    save_confusion_matrix(metrics["confusion_matrix"], OUT / "confusion_matrix", class_names=CLASS_NAMES)
    plot_pooling_topology(
        regions,
        test_out["attention"]["rhythm"][0].detach().numpy(),
        test_out["attention"]["region"][0].detach().numpy(),
        int(y_test[0]), int(test_prob[0].argmax()),
        OUT / "graph_topology",
    )

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
![Pooling attention on one real held-out window](graph_topology.svg)

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
