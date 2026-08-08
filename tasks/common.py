"""Shared real-STEW loading, subject-disjoint splitting, and plotting helpers
for the three task folders (graph_prediction, node_prediction, link_prediction).

Every task evaluates the same trained artifact -- hmb_kuramoto_ode.models.full_model
.HierarchicalKuramotoODE -- against the STEW recordings already checked into
dataset/. No synthetic fallback: if dataset/ is missing this raises, matching the
rest of the repository's real-data commands.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _repository_root = Path(__file__).resolve().parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from hmb_kuramoto_ode.data.graph_builder import batch_edges, build_hierarchical_graph
from hmb_kuramoto_ode.data.preprocessing import RhythmPreprocessor, TrainNormalizer
from hmb_kuramoto_ode.data.splits import assert_disjoint
from hmb_kuramoto_ode.data.stew import STEWDataset

FEATURE_NAMES = ("log_power", "relative_power", "amplitude", "phase_sin", "phase_cos", "spectral_entropy")


def load_stew_windows(data_root: str, windows_per_record: int = 3, sfreq: float = 128.0,
                       window_seconds: float = 4.0, overlap: float = 0.5) -> list[dict]:
    """Load a bounded number of rhythm-feature windows per STEW recording.

    Never substitutes synthetic data: STEWDataset raises if data_root has no
    hi/lo recordings.
    """
    preprocessor = RhythmPreprocessor(sfreq=sfreq, window_seconds=window_seconds, overlap=overlap)
    dataset = STEWDataset(data_root, preprocessor)
    rows = []
    for record in dataset.records:
        raw = dataset.load(record)
        for start, window in preprocessor.windows(raw)[:windows_per_record]:
            rows.append({
                "features": preprocessor.transform_window(window),
                "label": record.label,
                "subject": record.subject_id,
                "source": record.path.name,
                "start": start,
            })
    if len({row["subject"] for row in rows}) < 5:
        raise ValueError("real evaluation needs at least five subjects for a disjoint train/validation/test split")
    return rows


def limit_subjects(rows: list[dict], max_subjects: int | None, seed: int = 7) -> list[dict]:
    """Randomly keep at most max_subjects subjects, to bound runtime on CPU."""
    subjects = sorted({row["subject"] for row in rows})
    if max_subjects is None or max_subjects >= len(subjects):
        return rows
    order = np.random.default_rng(seed).permutation(len(subjects))
    keep = {subjects[i] for i in order[:max_subjects]}
    return [row for row in rows if row["subject"] in keep]


def subject_disjoint_split(rows: list[dict], n_test_subjects: int = 2, n_val_subjects: int = 2,
                            seed: int = 7) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """Split by subject, never by window, so no recording leaks across folds."""
    subjects = sorted({row["subject"] for row in rows})
    order = np.random.default_rng(seed).permutation(len(subjects))
    shuffled = [subjects[i] for i in order]
    test = set(shuffled[:n_test_subjects])
    validation = set(shuffled[n_test_subjects:n_test_subjects + n_val_subjects])
    train = set(subjects) - test - validation
    assert_disjoint(train, validation, test)
    split = {
        "train": [row for row in rows if row["subject"] in train],
        "validation": [row for row in rows if row["subject"] in validation],
        "test": [row for row in rows if row["subject"] in test],
    }
    subject_ids = {"train": sorted(train), "validation": sorted(validation), "test": sorted(test)}
    return split, subject_ids


def fit_and_transform(split: dict[str, list[dict]], device: str = "cpu"):
    """Fit TrainNormalizer on the training rows only, then build tensors for all folds."""
    train_features = np.stack([row["features"] for row in split["train"]])
    normalizer = TrainNormalizer().fit(train_features, [row["subject"] for row in split["train"]])
    tensors = {}
    for fold, rows in split.items():
        features = normalizer.transform(np.stack([row["features"] for row in rows]))
        x = torch.tensor(features, dtype=torch.float32, device=device)
        y = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
        tensors[fold] = (x, y)
    return tensors, normalizer


def hierarchical_edges(regions: int, batch_size: int):
    base_edges, edge_types = build_hierarchical_graph(regions)
    batched = batch_edges(base_edges, regions * 5, batch_size)
    return batched, base_edges, edge_types


def sample_link_pairs(base_edges: torch.Tensor, regions: int, batch_size: int, seed: int = 7):
    """Balanced positive/negative edges per sample, offset into a batched graph.

    Positives are the canonical (u < v) hierarchical edges -- same-band spatial
    and local cross-frequency links. Negatives are canonical non-edges: node
    pairs within the same window that the hierarchy never connects. Pairs never
    cross a window boundary, mirroring the "samples remain disconnected" batching
    contract in data/graph_builder.py.
    """
    nodes_per_graph = regions * 5
    positives = base_edges[:, 0::2]  # every add() call emits (u, v) then (v, u) with u < v
    positive_set = set(zip(positives[0].tolist(), positives[1].tolist()))
    rng = np.random.default_rng(seed)
    n_positive = positives.shape[1]
    all_pairs, all_labels = [], []
    for sample in range(batch_size):
        offset = sample * nodes_per_graph
        all_pairs.append(positives + offset)
        all_labels.append(torch.ones(n_positive))
        negatives, seen = [], set()
        while len(negatives) < n_positive:
            u, v = rng.integers(0, nodes_per_graph, size=2)
            if u == v:
                continue
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            if (a, b) in positive_set or (a, b) in seen:
                continue
            seen.add((a, b))
            negatives.append((a + offset, b + offset))
        all_pairs.append(torch.tensor(negatives, dtype=torch.long).T)
        all_labels.append(torch.zeros(n_positive))
    pairs = torch.cat(all_pairs, dim=1)
    labels = torch.cat(all_labels)
    return pairs, labels


def plot_loss_curve(history: list[dict], path, secondary_key: str | None = None, secondary_label: str | None = None):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="validation loss")
    ax.set(xlabel="epoch", ylabel="loss")
    handles, labels = ax.get_legend_handles_labels()
    if secondary_key:
        ax2 = ax.twinx()
        ax2.plot(epochs, [h[secondary_key] for h in history], color="darkorange", linestyle="--",
                  label=secondary_label or secondary_key)
        ax2.set_ylabel(secondary_label or secondary_key)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_roc_pr(y_true, scores, path, title: str = ""):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color="#2563eb")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"{title} ROC")
    axes[1].plot(recall, precision, color="#2563eb")
    axes[1].set(xlabel="recall", ylabel="precision", title=f"{title} PR")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(y_true, y_pred, s=5, alpha=0.35, color="#2563eb")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set(xlabel="true (normalized feature value)", ylabel="predicted", title="Predicted vs true")
    residual = y_pred - y_true
    axes[1].hist(residual, bins=40, color="#2563eb")
    axes[1].set(xlabel="residual (predicted - true)", ylabel="count", title="Residual distribution")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_feature_errors(mae_per_feature: list[float], path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(FEATURE_NAMES, mae_per_feature, color="#2563eb")
    ax.set(ylabel="MAE (normalized units)", title="Masked-feature reconstruction error by feature")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
