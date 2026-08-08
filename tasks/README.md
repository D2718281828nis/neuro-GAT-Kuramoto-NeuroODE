# Three-task evaluation suite

This folder evaluates one architecture — [`HierarchicalKuramotoODE`](../hmb_kuramoto_ode/models/full_model.py)
— against the three prediction problems its multi-task heads are built for, using
real STEW recordings (`../dataset/`, subject-disjoint, no synthetic fallback):

| Task | Folder | Head exercised | Metric family |
|---|---|---|---|
| Graph prediction (workload hi/lo) | [`graph_prediction/`](graph_prediction/TASK.md) | `graph_head` | accuracy, ROC-AUC, confusion matrix |
| Node prediction (masked-feature reconstruction) | [`node_prediction/`](node_prediction/TASK.md) | `node_head` | MAE, RMSE, R² |
| Link prediction (hierarchical edge reconstruction) | [`link_prediction/`](link_prediction/TASK.md) | `link_head` | ROC-AUC, average precision, confusion matrix |

Each task folder has:

- **`TASK.md`** — what the task is, exactly which modules/heads/losses solve it
  (with file references into `hmb_kuramoto_ode/`), the evaluation protocol, and
  the measured results.
- **`evaluate.py`** — a standalone script: loads real STEW windows, splits by
  subject, trains the relevant path of the model, evaluates on held-out
  subjects, and writes `results/results.json`, `results/results.md`, and PNG/SVG
  plots.

`common.py` in this folder holds the logic all three scripts share: STEW window
loading, subject-disjoint splitting (`hmb_kuramoto_ode/data/splits.py`), tensor
normalization (fit on training subjects only, via `TrainNormalizer`), batched
hierarchical edges, balanced positive/negative link sampling, and the
matplotlib/NetworkX plotting helpers.

Every task also writes `results/graph_topology.svg`: a NetworkX drawing of the
same 70-node (14 electrode x 5 band) hierarchical graph every task shares,
laid out at approximate scalp positions, with a task-specific overlay drawn
from a real forward pass on a real held-out window -- learned pooling
attention for graph prediction, masked reconstruction targets for node
prediction, and per-edge correct/wrong outcomes for link prediction. See each
`TASK.md` for exactly what is encoded in its version.

## How this differs from the other reports in the repository

- [`reports/synthetic_metrics.md`](../reports/synthetic_metrics.md) exercises the
  *reporting code* against fixed, hand-written fixtures — it never runs the
  model. This suite runs the actual `HierarchicalKuramotoODE` forward/backward
  pass end to end.
- [`reports/real_stew_status.md`](../reports/real_stew_status.md) and
  `examples/stew_real_experiment.py` already cover real-data **graph**
  prediction (GAT baseline vs. full model). This suite adds real-data **node**
  and **link** evaluation, which didn't previously exist, and gives graph
  prediction its own dedicated writeup and plots.

## Honesty notes

- Every script trains from scratch on a bounded subject subset
  (`--max-subjects`, default 12 of the 48 available) and a modest epoch count,
  so a full run finishes in roughly a minute on a CPU. This is a smoke-scale
  demonstration of the real architecture on real data, not the
  subject-independent, cross-validated protocol `configs/stew_full.yaml` and
  `hmb_kuramoto_ode.cli cross-validate` are built for. Widen `--max-subjects`
  and `--epochs` for a less noisy estimate.
- No script substitutes synthetic data if `dataset/` is missing or malformed;
  they fail the same way `hmb_kuramoto_ode.cli` and `stew_real_experiment.py`
  do.
- Subject IDs used for train/validation/test are written into every
  `results/results.json`, so leakage can be checked directly.

## Run everything

```bash
python tasks/graph_prediction/evaluate.py --data-root dataset
python tasks/node_prediction/evaluate.py --data-root dataset
python tasks/link_prediction/evaluate.py --data-root dataset
```
