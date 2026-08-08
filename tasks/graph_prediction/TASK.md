# Task 1 — Graph prediction (window-level workload classification)

## What the task is

Given one STEW window — the full hierarchical rhythm graph for a subject at one
moment (14 electrodes × 5 bands × 6 features) — predict the binary label already
attached to that window's recording: `high` (SIMKAP multitasking) or `low`
(resting) workload. This is a whole-graph classification task: one label per
graph, not per node or per edge.

## How `hmb_kuramoto_ode` solves it

1. **Encode.** [`models/encoder.py`](../../hmb_kuramoto_ode/models/encoder.py) turns the raw `[B, 14, 5, 6]`
   rhythm features into oscillator states `[B, 70, hidden]`: a band-specific linear
   projection, a learned region (electrode) embedding, and a residual feature
   projection are summed, and latent coordinate 0 is overwritten with
   `atan2(sin_phase, cos_phase)` so it is a genuine phase angle rather than a
   generic feature.
2. **Integrate.** [`models/ode_func.py`](../../hmb_kuramoto_ode/models/ode_func.py)'s `KuramotoGrandVectorField` computes
   edge-normalized attention (`models/attention.py`), a bounded Kuramoto phase-coupling
   term restricted to coordinate 0 (spatial same-band edges, plus optional local
   cross-frequency edges), and a GRAND-style diffusion term over every coordinate.
   A learned sigmoid gate mixes the two per node. [`models/ode_solver.py`](../../hmb_kuramoto_ode/models/ode_solver.py)
   integrates this vector field with fixed-step RK4 from `t=0` to `t=t1`, so the whole
   window's synchronization dynamics are literally simulated, not just filtered.
3. **Pool.** [`models/pooling.py`](../../hmb_kuramoto_ode/models/pooling.py)'s `HierarchicalAttentionPooling` first
   attention-pools the 5 band nodes into each electrode, then attention-pools the 14
   electrodes into one graph embedding, masking any missing electrodes before the
   softmax.
4. **Classify.** [`models/full_model.py`](../../hmb_kuramoto_ode/models/full_model.py)'s `graph_head` is a single
   `nn.Linear(hidden, 2)` on top of the pooled graph embedding, trained with
   cross-entropy against the hi/lo label.

Everything upstream of `graph_head` — attention, coupling strength, gate, and the
RK4 trajectory itself — receives gradient from this loss, so the synchronization
dynamics the model discovers are shaped directly by whether they help separate
the two workload conditions.

## Data and evaluation protocol

`tasks/common.py` loads real STEW recordings from `dataset/` (never synthetic),
windows them with `RhythmPreprocessor` (4 s windows, 50% overlap, 128 Hz), and
splits by **subject**, not by window: a `TrainNormalizer` is fit only on training
subjects, and `data/splits.py::assert_disjoint` guards against any subject
appearing in more than one fold. This mirrors `examples/stew_real_experiment.py`
and the leakage-prevention rules documented in the top-level `README.md`.

Metric definitions come straight from [`training/metrics.py`](../../hmb_kuramoto_ode/training/metrics.py):
accuracy, balanced accuracy, macro-F1, ROC-AUC, AUPRC, and expected calibration
error (ECE), plus a confusion matrix rendered by
`analysis/visualization.py::save_confusion_matrix`. Model selection uses
validation balanced accuracy; the reported numbers are test-subject only.

## Results

See [`results/results.md`](results/results.md) (and `results/results.json` for the
raw numbers) for the latest measured run, plus `results/loss_curve.svg`,
`results/roc_pr.svg`, and `results/confusion_matrix.svg`.

`results/graph_topology.svg` is a NetworkX drawing of the actual hierarchical
graph for one real held-out test window: the 14 electrodes are laid out in
their approximate scalp positions, each electrode's 5 band nodes are drawn in
a small ring around it and colored by band, and node size is that node's
*real* `rhythm` attention weight from `HierarchicalAttentionPooling` on that
window. Orange lines run from each electrode to a central "graph embedding"
node, with line width equal to that electrode's `region` attention weight --
i.e. how much it contributes to the vector `graph_head` classifies. This is
not a schematic; the weights are read directly off a trained forward pass.

The default configuration below only uses a subset of STEW subjects
(`--max-subjects 12`) so the demonstration finishes in about a minute on a CPU;
treat these as an honest small-scale smoke run of the real architecture and
real data, not the full-protocol scientific result that `configs/stew_full.yaml`
and `cross-validate` are built for.

## Reproduce

```bash
python tasks/graph_prediction/evaluate.py --data-root dataset
# more subjects / epochs for a less noisy estimate:
python tasks/graph_prediction/evaluate.py --data-root dataset --max-subjects 30 --epochs 40
```
