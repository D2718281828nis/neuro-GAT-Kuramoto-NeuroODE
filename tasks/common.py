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

from hmb_kuramoto_ode.contracts import BANDS, DEFAULT_CHANNELS
from hmb_kuramoto_ode.data.graph_builder import (
    CROSS_FREQUENCY_LOCAL,
    SPATIAL_SAME_BAND,
    batch_edges,
    build_hierarchical_graph,
)
from hmb_kuramoto_ode.data.preprocessing import RhythmPreprocessor, TrainNormalizer
from hmb_kuramoto_ode.data.splits import assert_disjoint
from hmb_kuramoto_ode.data.stew import STEWDataset

FEATURE_NAMES = ("log_power", "relative_power", "amplitude", "phase_sin", "phase_cos", "spectral_entropy")

# Approximate top-down 10-20-style layout for the 14 Emotiv EPOC channels in
# contracts.DEFAULT_CHANNELS order, used only for a readable schematic -- not
# a claim of anatomically precise electrode coordinates.
ELECTRODE_POSITIONS = {
    "AF3": (-0.30, 0.90), "AF4": (0.30, 0.90),
    "F7": (-0.80, 0.55), "F3": (-0.35, 0.55), "F4": (0.35, 0.55), "F8": (0.80, 0.55),
    "FC5": (-0.65, 0.20), "FC6": (0.65, 0.20),
    "T7": (-1.00, -0.15), "T8": (1.00, -0.15),
    "P7": (-0.80, -0.55), "P8": (0.80, -0.55),
    "O1": (-0.30, -0.90), "O2": (0.30, -0.90),
}
BAND_COLORS = {"delta": "#7c3aed", "theta": "#2563eb", "alpha": "#0891b2", "beta": "#059669", "gamma": "#d97706"}


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


# --- NetworkX visualizations of the hierarchical electrode/rhythm graph -----
#
# One 70-node graph underlies every task: 14 electrodes x 5 rhythm bands,
# connected by the same-band spatial and local cross-frequency edges from
# data/graph_builder.py::build_hierarchical_graph. Each task highlights a
# different part of how it uses that graph: pooling (graph_prediction),
# masking (node_prediction), or the edge/non-edge decision itself
# (link_prediction).

def hierarchical_layout(regions: int, cluster_radius: float = 0.16) -> dict[int, tuple[float, float]]:
    """Node position for every (electrode, band) node: band nodes sit in a small
    ring around their electrode's approximate scalp position."""
    positions = {}
    for electrode, name in enumerate(DEFAULT_CHANNELS[:regions]):
        cx, cy = ELECTRODE_POSITIONS[name]
        for band in range(len(BANDS)):
            angle = 2 * np.pi * band / len(BANDS)
            node = electrode * len(BANDS) + band
            positions[node] = (cx + cluster_radius * np.cos(angle), cy + cluster_radius * np.sin(angle))
    return positions


def build_networkx_graph(regions: int):
    """The same hierarchical graph the model trains on, as an undirected
    networkx.Graph with one entry per canonical (u < v) edge and its type."""
    import networkx as nx

    edge_index, edge_types = build_hierarchical_graph(regions)
    graph = nx.Graph()
    graph.add_nodes_from(range(regions * len(BANDS)))
    for (u, v), edge_type in zip(edge_index[:, 0::2].T.tolist(), edge_types[0::2].tolist()):
        graph.add_edge(u, v, type=edge_type)
    return graph


def draw_hierarchical_topology(ax, regions: int, node_size=60, alpha=1.0):
    """Draw the base graph (band-colored nodes, typed edges, electrode labels)
    onto an existing matplotlib Axes and return its node -> (x, y) layout so
    callers can add task-specific highlights on top."""
    import networkx as nx

    graph = build_networkx_graph(regions)
    pos = hierarchical_layout(regions)
    spatial = [(u, v) for u, v, d in graph.edges(data=True) if d["type"] == SPATIAL_SAME_BAND]
    cross_frequency = [(u, v) for u, v, d in graph.edges(data=True) if d["type"] == CROSS_FREQUENCY_LOCAL]
    nx.draw_networkx_edges(graph, pos, edgelist=spatial, edge_color="#cbd5e1", width=0.7, ax=ax, alpha=alpha)
    nx.draw_networkx_edges(graph, pos, edgelist=cross_frequency, edge_color="#e2e8f0", width=0.5, style=":", ax=ax, alpha=alpha)
    node_colors = [BAND_COLORS[BANDS[node % len(BANDS)]] for node in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_size, ax=ax,
                            linewidths=0.4, edgecolors="white", alpha=alpha)
    for electrode, name in enumerate(DEFAULT_CHANNELS[:regions]):
        cx, cy = ELECTRODE_POSITIONS[name]
        ax.text(cx, cy + 0.20, name, ha="center", fontsize=7.5, fontweight="bold", color="#334155")
    return graph, pos


def band_legend_handles():
    import matplotlib.pyplot as plt

    return [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=band)
            for band, color in BAND_COLORS.items()]


def save_topology_figure(fig, path, title: str):
    import matplotlib.pyplot as plt

    fig.gca().set(title=title)
    fig.gca().set_aspect("equal")
    fig.gca().axis("off")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
