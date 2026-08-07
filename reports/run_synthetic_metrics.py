"""Create deterministic three-task smoke metrics without external dependencies.

These fixtures validate reporting code only; they are not STEW/model benchmark results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

OUT = Path(__file__).parent


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Binary ROC-AUC via pairwise concordance, with 0.5 credit for ties."""
    positive = [s for y, s in zip(labels, scores) if y == 1]
    negative = [s for y, s in zip(labels, scores) if y == 0]
    if not positive or not negative:
        raise ValueError("ROC-AUC requires both positive and negative examples")
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> list[list[int]]:
    """Return rows=true label and columns=predicted label: [[TN, FP], [FN, TP]]."""
    matrix = [[0, 0], [0, 0]]
    for target, score in zip(labels, scores):
        matrix[target][int(score >= threshold)] += 1
    return matrix


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, object]:
    matrix = confusion(labels, scores)
    correct = matrix[0][0] + matrix[1][1]
    return {
        "samples": len(labels),
        "threshold": 0.5,
        "roc_auc": roc_auc(labels, scores),
        "accuracy": correct / len(labels),
        "confusion_matrix": matrix,
        "confusion_matrix_axes": {"rows": "true [0, 1]", "columns": "predicted [0, 1]"},
    }


def regression_metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, object]:
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    mean_target = sum(targets) / len(targets)
    residual = sum(error * error for error in errors)
    total = sum((target - mean_target) ** 2 for target in targets)
    return {
        "values": len(targets),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(residual / len(errors)),
        "r2": 1.0 - residual / total,
        "roc_auc": None,
        "confusion_matrix": None,
        "classification_metrics_note": "Not applicable to continuous masked-feature reconstruction.",
    }


def matrix_svg(title: str, matrix: list[list[int]], x: int, y: int) -> str:
    maximum = max(max(row) for row in matrix) or 1
    cells = []
    for row in range(2):
        for column in range(2):
            value = matrix[row][column]
            opacity = 0.2 + 0.75 * value / maximum
            cells.append(
                f'<rect x="{x + column*105}" y="{y + row*85}" width="100" height="80" '
                f'fill="#2563eb" fill-opacity="{opacity:.3f}" stroke="#1e293b"/>'
                f'<text x="{x + column*105 + 50}" y="{y + row*85 + 48}" class="value">{value}</text>'
            )
    return (
        f'<text x="{x+102}" y="{y-42}" class="title">{title}</text>'
        f'<text x="{x+102}" y="{y-18}" class="small">columns: predicted 0 / 1</text>'
        + ''.join(cells)
        + f'<text x="{x-15}" y="{y+82}" class="small" transform="rotate(-90 {x-15} {y+82})">rows: true 0 / 1</text>'
    )


def write_visualization(metrics: dict[str, object]) -> None:
    graph = metrics["graph_prediction"]
    link = metrics["link_prediction"]
    node = metrics["node_prediction"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520" role="img" aria-labelledby="title desc">
<title id="title">Synthetic three-task smoke metrics</title>
<desc id="desc">Graph and link confusion matrices plus node regression metrics. These are deterministic fixture results, not STEW results.</desc>
<style>.title{{font:700 20px sans-serif;fill:#172033;text-anchor:middle}}.small{{font:14px sans-serif;fill:#475569;text-anchor:middle}}.value{{font:700 26px sans-serif;fill:white;text-anchor:middle}}.metric{{font:600 18px sans-serif;fill:#172033}}</style>
<rect width="1000" height="520" fill="#f8fafc"/>
<text x="500" y="35" class="title">Synthetic smoke evaluation — not a STEW benchmark</text>
{matrix_svg(f"Graph prediction — ROC-AUC {graph['roc_auc']:.3f}", graph['confusion_matrix'], 90, 135)}
{matrix_svg(f"Link prediction — ROC-AUC {link['roc_auc']:.3f}", link['confusion_matrix'], 400, 135)}
<rect x="700" y="92" width="255" height="235" rx="8" fill="#ecfeff" stroke="#155e75" stroke-width="2"/>
<text x="827" y="130" class="title">Node prediction</text>
<text x="725" y="180" class="metric">MAE: {node['mae']:.4f}</text>
<text x="725" y="220" class="metric">RMSE: {node['rmse']:.4f}</text>
<text x="725" y="260" class="metric">R²: {node['r2']:.4f}</text>
<text x="827" y="300" class="small">ROC-AUC / matrix: N/A (regression)</text>
<text x="500" y="445" class="small">Rows are true classes; columns are predicted classes; classification threshold = 0.5.</text>
<text x="500" y="478" class="small">Fixed synthetic fixtures exercise report calculations only. No dataset or trained checkpoint was used.</text>
</svg>'''
    (OUT / "synthetic_metrics.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    # Fixed fixtures make the smoke report deterministic and auditable.
    graph_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    graph_scores = [0.08, 0.18, 0.32, 0.41, 0.62, 0.27, 0.38, 0.57, 0.68, 0.79, 0.88, 0.93]
    link_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    link_scores = [0.10, 0.91, 0.22, 0.82, 0.35, 0.73, 0.58, 0.64, 0.16, 0.48, 0.44, 0.77, 0.30, 0.69, 0.55, 0.87]
    node_targets = [-1.0, -0.7, -0.2, 0.0, 0.25, 0.5, 0.9, 1.2, 1.5, 1.8]
    node_predictions = [-0.9, -0.82, -0.1, 0.08, 0.18, 0.62, 0.75, 1.28, 1.37, 1.93]
    metrics = {
        "report_type": "deterministic synthetic smoke fixture",
        "stew_result": False,
        "graph_prediction": classification_metrics(graph_labels, graph_scores),
        "node_prediction": regression_metrics(node_targets, node_predictions),
        "link_prediction": classification_metrics(link_labels, link_scores),
    }
    (OUT / "synthetic_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_visualization(metrics)
    graph = metrics["graph_prediction"]
    node = metrics["node_prediction"]
    link = metrics["link_prediction"]
    markdown = f"""# Three-task synthetic smoke metrics

> **Scope:** measured from deterministic fixtures in `run_synthetic_metrics.py`; these are not STEW results and were not produced by a trained checkpoint.

![Three-task metrics](synthetic_metrics.svg)

## 1. Node prediction — masked-feature reconstruction

This is a continuous regression task, so ROC-AUC and a confusion matrix are not mathematically applicable without inventing a thresholded label.

| Metric | Measured value |
|---|---:|
| Values | {node['values']} |
| MAE | {node['mae']:.6f} |
| RMSE | {node['rmse']:.6f} |
| R² | {node['r2']:.6f} |
| ROC-AUC | N/A — regression |
| Confusion matrix | N/A — regression |

## 2. Link prediction — binary edge reconstruction

| Metric | Measured value |
|---|---:|
| Edges | {link['samples']} |
| ROC-AUC | {link['roc_auc']:.6f} |
| Accuracy at 0.5 | {link['accuracy']:.6f} |

Confusion matrix (`rows=true [0,1]`, `columns=predicted [0,1]`):

```text
{link['confusion_matrix'][0]}
{link['confusion_matrix'][1]}
```

## 3. Graph prediction — binary graph classification

| Metric | Measured value |
|---|---:|
| Graphs | {graph['samples']} |
| ROC-AUC | {graph['roc_auc']:.6f} |
| Accuracy at 0.5 | {graph['accuracy']:.6f} |

Confusion matrix (`rows=true [0,1]`, `columns=predicted [0,1]`):

```text
{graph['confusion_matrix'][0]}
{graph['confusion_matrix'][1]}
```

## Reproduce

```bash
python reports/run_synthetic_metrics.py
```

The command uses only the Python standard library and deterministically rewrites this Markdown file, `synthetic_metrics.json`, and the text-based `synthetic_metrics.svg` visualization. Real STEW ROC-AUC and confusion matrices must be generated from held-out subjects and must not be inferred from this smoke fixture.
"""
    (OUT / "synthetic_metrics.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
