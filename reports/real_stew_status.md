# Real STEW execution status

The real-data runner is available, but **no real metric is committed from this environment**: the referenced [`dataset/`](https://github.com/D2718281828nis/neuro-GAT-Kuramoto-NeuroODE/tree/main/dataset) could not be downloaded because every GitHub, raw-content, API, and codeload request was rejected by the execution proxy with HTTP 403, no `dataset/` files are mounted locally, and the supplied Python 3.14 runtime lacks NumPy/PyTorch. The attempted GAT command therefore stopped at dependency import before reading data. Reporting a ROC-AUC or confusion matrix under those conditions would fabricate a result.

Once the repository dataset is locally available, run subject-disjoint GAT training and the full architecture separately:

```bash
python -m examples.stew_real_experiment --data-root dataset --model gat --epochs 20
python -m examples.stew_real_experiment --data-root dataset --model full --epochs 20
```

Each command refuses synthetic fallback and writes `reports/real_stew/<model>_metrics.json`, `<model>_report.md`, and both PNG/SVG training-loss plus validation-accuracy plots. Dataset files remain ignored and must not be committed.
