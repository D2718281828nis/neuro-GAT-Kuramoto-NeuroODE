# Task 3 — Link prediction (hierarchical edge reconstruction)

## What the task is

Given the window-conditioned node embeddings `z(t1)` after ODE integration,
decide for each candidate pair of nodes whether the hierarchical graph
actually connects them. Positive pairs are canonical (`u < v`) edges from
[`data/graph_builder.py`](../../hmb_kuramoto_ode/data/graph_builder.py): same-band spatial edges between
neighboring electrodes, and local cross-frequency edges between the 5 rhythms
inside one electrode. Negative pairs are canonical non-edges within the same
window — node pairs the hierarchy never connects, sampled without duplicates
or reversals, matching the negative-sampling contract described in the
top-level `README.md`.

One subtlety worth stating plainly: which pairs are positive/negative is fixed
by the electrode/band topology and does not change from window to window. What
*does* change per window is `z(t1)` — a real function of that window's EEG
features, integrated through subject- and moment-specific Kuramoto/GRAND
dynamics. So this task measures whether the representation the model builds
from real EEG dynamics retains enough structural information to reconstruct
the known graph topology, evaluated on subjects the model never trained on.

## How `hmb_kuramoto_ode` solves it

Encode and integrate exactly as in [Task 1](../graph_prediction/TASK.md) and
[Task 2](../node_prediction/TASK.md). The link-specific piece is
[`models/heads.py`](../../hmb_kuramoto_ode/models/heads.py)'s `LinkDecoder`: for a candidate edge `(u, v)`
it builds `z_u || z_v || |z_u - z_v| || z_u * z_v` — concatenation, absolute
difference, and elementwise product of the two node embeddings — and passes
that through a two-layer MLP to a single logit, trained with
`binary_cross_entropy_with_logits` against the edge/non-edge label. This
feature combination lets the decoder use both similarity (product,
difference) and raw identity (concatenation) of the two endpoints, which
matters here because a true edge's two endpoints can have very different
raw phases (e.g. delta vs. gamma at the same electrode) despite being
structurally connected.

`[edge_pairs]` is threaded through [`models/full_model.py`](../../hmb_kuramoto_ode/models/full_model.py)'s
`forward` as an optional argument, so the same trained checkpoint can serve
graph, node, and link predictions from one integration pass; this task
exercises that argument directly.

## Data and evaluation protocol

Same real-STEW, subject-disjoint pipeline as Tasks 1–2. `tasks/common.py`'s
`sample_link_pairs` builds a balanced positive/negative set per window,
offset into the batched multi-window graph so no edge crosses a window
boundary (mirroring the "samples remain disconnected" batching contract in
`data/graph_builder.py::batch_edges`). Metrics — ROC-AUC and average
precision — come from [`training/metrics.py`](../../hmb_kuramoto_ode/training/metrics.py)`::link_metrics`;
accuracy at a 0.5 probability threshold and a confusion matrix are computed
alongside for readability. Model selection uses validation ROC-AUC.

## Results

See [`results/results.md`](results/results.md) (and `results/results.json`) for the
latest measured ROC-AUC, average precision, and accuracy, plus
`results/loss_curve.svg`, `results/roc_pr.svg`, and
`results/confusion_matrix.svg`.

`results/graph_topology.svg` is a NetworkX drawing of a subsample of one held-out
test window's candidate pairs directly on the real hierarchical graph: solid
lines are true edges, dashed lines are true non-edges, and color is whether
`LinkDecoder` got that specific pair right (green) or wrong (red) on that
window. It is the literal per-edge outcome of the trained decoder, not a
schematic of the topology alone.

As with Tasks 1–2, the default run uses a bounded subject subset for a fast
CPU demonstration; it is a smoke evaluation of the real model on real data,
not the full-protocol result.

## Reproduce

```bash
python tasks/link_prediction/evaluate.py --data-root dataset
```
