"""Node prediction: reconstruct masked (electrode, rhythm) feature vectors.

A random subset of nodes has its six-dimensional rhythm feature vector zeroed
before encoding; models/heads.py's node_head must recover the original,
normalized feature vector purely from the surviving neighbors propagated
through the Kuramoto/GRAND vector field. Evaluated on real, subject-disjoint
STEW windows. See TASK.md in this folder for the write-up.

Usage:
    python tasks/node_prediction/evaluate.py --data-root dataset
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

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.utils.seed import seed_everything
from tasks.common import (
    FEATURE_NAMES,
    fit_and_transform,
    hierarchical_edges,
    limit_subjects,
    load_stew_windows,
    plot_feature_errors,
    plot_loss_curve,
    plot_regression,
    subject_disjoint_split,
)

OUT = Path(__file__).parent / "results"


def node_mask(batch_size: int, regions: int, bands: int, fraction: float, seed: int) -> torch.Tensor:
    """Boolean [batch, regions, bands] mask, a fixed fraction of nodes True per sample."""
    n_nodes = regions * bands
    k = max(1, int(fraction * n_nodes))
    rng = np.random.default_rng(seed)
    mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool)
    for sample in range(batch_size):
        idx = rng.choice(n_nodes, size=k, replace=False)
        mask[sample, idx] = True
    return mask.reshape(batch_size, regions, bands)


def masked_mse(prediction, target, mask):
    diff = (prediction - target)[mask]
    return diff.pow(2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--max-subjects", type=int, default=12)
    parser.add_argument("--windows-per-record", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mask-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed)

    rows = limit_subjects(load_stew_windows(args.data_root, args.windows_per_record), args.max_subjects, args.seed)
    split, subject_ids = subject_disjoint_split(rows, seed=args.seed)
    tensors, _ = fit_and_transform(split)
    x_train, _ = tensors["train"]
    x_val, _ = tensors["validation"]
    x_test, _ = tensors["test"]
    regions, bands = x_train.shape[1], x_train.shape[2]

    val_mask = node_mask(x_val.shape[0], regions, bands, args.mask_fraction, args.seed + 1)
    test_mask = node_mask(x_test.shape[0], regions, bands, args.mask_fraction, args.seed + 2)

    base_edges, _, _ = hierarchical_edges(regions, 1)
    model = HierarchicalKuramotoODE(base_edges, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def set_edges(batch_size):
        edges, _, _ = hierarchical_edges(regions, batch_size)
        model.field.edge_index = edges

    history = []
    best_state, best_val_loss = None, float("inf")
    for epoch in tqdm(range(1, args.epochs + 1), desc="node_prediction", unit="epoch"):
        model.train()
        set_edges(x_train.shape[0])
        train_mask = node_mask(x_train.shape[0], regions, bands, args.mask_fraction, args.seed + 1000 + epoch)
        x_masked = x_train * (~train_mask).unsqueeze(-1)
        optimizer.zero_grad()
        prediction = model(x_masked)["node_prediction"]
        loss = masked_mse(prediction, x_train, train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            set_edges(x_val.shape[0])
            x_val_masked = x_val * (~val_mask).unsqueeze(-1)
            val_prediction = model(x_val_masked)["node_prediction"]
            val_loss = float(masked_mse(val_prediction, x_val, val_mask))
        history.append({"epoch": epoch, "train_loss": float(loss), "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("edge_index")}

    model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        set_edges(x_test.shape[0])
        x_test_masked = x_test * (~test_mask).unsqueeze(-1)
        test_prediction = model(x_test_masked)["node_prediction"]

    true_masked = x_test[test_mask].numpy()
    pred_masked = test_prediction[test_mask].numpy()
    mae = float(mean_absolute_error(true_masked, pred_masked))
    rmse = float(mean_squared_error(true_masked, pred_masked) ** 0.5)
    r2 = float(r2_score(true_masked, pred_masked))
    mae_per_feature = [
        float(mean_absolute_error(x_test[test_mask][:, i].numpy(), test_prediction[test_mask][:, i].detach().numpy()))
        for i in range(len(FEATURE_NAMES))
    ]
    variance_per_feature = [float(x_test[test_mask][:, i].numpy().var()) for i in range(len(FEATURE_NAMES))]
    low_variance_features = [name for name, v in zip(FEATURE_NAMES, variance_per_feature) if v < 1e-6]

    OUT.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, OUT / "loss_curve")
    plot_regression(true_masked.reshape(-1), pred_masked.reshape(-1), OUT / "predicted_vs_true")
    plot_feature_errors(mae_per_feature, OUT / "feature_errors")

    metrics = {
        "masked_values": int(true_masked.size),
        "masked_nodes": int(test_mask.sum()),
        "mask_fraction": args.mask_fraction,
        "mae": mae, "rmse": rmse, "r2": r2,
        "mae_per_feature": dict(zip(FEATURE_NAMES, mae_per_feature)),
        "variance_per_feature": dict(zip(FEATURE_NAMES, variance_per_feature)),
        "low_variance_features": low_variance_features,
    }
    payload = {
        "task": "node_prediction",
        "dataset": "real STEW recordings under " + str(Path(args.data_root).resolve()),
        "subjects": subject_ids,
        "windows": {fold: len(rows_) for fold, rows_ in split.items()},
        "args": vars(args),
        "history": history,
        "test_metrics": metrics,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    feature_table = "\n".join(f"| {name} | {value:.4f} |" for name, value in metrics["mae_per_feature"].items())
    r2_note = ""
    if low_variance_features:
        r2_note = (
            f"\n> **R² caveat:** {', '.join(low_variance_features)} has near-zero variance in this small "
            "test set after normalization, so its true-value variance in the R² denominator is close to "
            "zero and a handful of small absolute errors produce a very large negative per-feature R². "
            "The aggregate R² above inherits that instability. MAE and RMSE, which do not divide by "
            "target variance, are the reliable headline numbers at this smoke scale.\n"
        )
    (OUT / "results.md").write_text(f"""# Node prediction — measured results

> Real STEW recordings from `{args.data_root}`, subject-disjoint split. Not a synthetic fixture.
> Regression task: ROC-AUC and a confusion matrix are not defined for continuous reconstruction.
{r2_note}
Subjects: train={subject_ids['train']}, validation={subject_ids['validation']}, test={subject_ids['test']}
Windows: train={len(split['train'])}, validation={len(split['validation'])}, test={len(split['test'])}
Masked nodes: {metrics['masked_nodes']} of {test_mask.numel()} ({args.mask_fraction:.0%} mask rate)

| Metric | Test value |
|---|---:|
| MAE | {mae:.4f} |
| RMSE | {rmse:.4f} |
| R² | {r2:.4f} |

MAE by feature (normalized units):

| Feature | MAE |
|---|---:|
{feature_table}

![Loss curve](loss_curve.svg)
![Predicted vs true, residuals](predicted_vs_true.svg)
![MAE by feature](feature_errors.svg)

## Reproduce

```bash
python tasks/node_prediction/evaluate.py --data-root {args.data_root}
```
""")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
