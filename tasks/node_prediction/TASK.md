# Task 2 — Node prediction (masked rhythm-feature reconstruction)

## What the task is

For every window, pick a random subset of the 70 `(electrode, band)` nodes
(30% by default) and zero out their entire 6-dimensional feature vector — log
power, relative power, analytic amplitude, phase sine/cosine, spectral entropy
— before the encoder ever sees it. The task is to reconstruct the original,
normalized feature vector at exactly those masked positions. This is a
continuous, self-supervised, per-node regression task, analogous to masked
feature modeling: it tests whether the learned dynamics can propagate enough
information from a node's neighbors — same-band electrodes and same-electrode
rhythms — to recover what was hidden.

## How `hmb_kuramoto_ode` solves it

The forward path is identical to the graph task through integration: encode
masked input with [`models/encoder.py`](../../hmb_kuramoto_ode/models/encoder.py), integrate the
Kuramoto/GRAND vector field with [`models/ode_solver.py`](../../hmb_kuramoto_ode/models/ode_solver.py). A masked
node's phase and features can only recover through the two coupling terms in
[`models/ode_func.py`](../../hmb_kuramoto_ode/models/ode_func.py): the Kuramoto sine-coupling term that pulls its phase
toward attention-weighted neighbor phases, and the GRAND diffusion term
`alpha * (h[src] - h[dst])` that pulls its full latent vector toward
attention-weighted neighbors along same-band spatial and local cross-frequency
edges. What differs from the graph task is the head and the loss:
[`models/full_model.py`](../../hmb_kuramoto_ode/models/full_model.py)'s `node_head` is a per-node
`nn.Linear(hidden, features)` applied to *every* node's post-integration state
`shaped = z.reshape(b, r, nb, -1)`, and `tasks/node_prediction/evaluate.py`
restricts the MSE loss to the masked positions only (`masked_mse` in that
file) — unmasked nodes are free context, not supervision targets, so the
model cannot simply copy its own input.

Because ROC-AUC and a confusion matrix require a thresholded binary label,
neither applies here; MAE, RMSE, and R² are the right metrics for this
regression task (as `reports/synthetic_metrics.md` also notes for its
fixture version of this task).

## Data and evaluation protocol

Same real-STEW, subject-disjoint pipeline as [Task 1](../graph_prediction/TASK.md), via
`tasks/common.py`. The node mask is regenerated every training epoch (a fresh
random 30% of nodes each time, seeded per-epoch) so the model cannot memorize
a fixed masking pattern; the validation and test masks are each generated once
with a fixed seed so the reported numbers are reproducible. Model selection
uses validation masked-MSE.

## Results

See [`results/results.md`](results/results.md) (and `results/results.json`) for the
latest measured MAE/RMSE/R² overall and broken down per feature, plus
`results/loss_curve.svg`, `results/predicted_vs_true.svg` (scatter + residual
histogram), and `results/feature_errors.svg` (MAE by feature).

As with Task 1, the default run uses a bounded subject subset for a fast CPU
demonstration; it is a smoke evaluation of the real model on real data, not
the full-protocol result.

## Reproduce

```bash
python tasks/node_prediction/evaluate.py --data-root dataset
# a stricter masking rate:
python tasks/node_prediction/evaluate.py --data-root dataset --mask-fraction 0.5
```
