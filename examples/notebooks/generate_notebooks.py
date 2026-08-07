"""Regenerates the Colab notebooks in this directory from the cell sources below.

Run with ``python examples/notebooks/generate_notebooks.py`` after editing any
of the NBxx_* cell strings, then re-execute the changed notebook(s) (for
example with ``jupyter nbconvert --to notebook --execute --inplace``) so the
committed .ipynb files keep real, checked outputs rather than stale or empty
ones. Mirrors the pattern used by reports/generate_architecture.py.
"""
import json
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/D2718281828nis/neuro-GAT-Kuramoto-NeuroODE.git"
REPO_BLOB = "https://github.com/D2718281828nis/neuro-GAT-Kuramoto-NeuroODE/blob/main"


def M(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def C(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def badge(filename):
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/D2718281828nis/neuro-GAT-Kuramoto-NeuroODE"
        f"/blob/main/examples/notebooks/{filename})"
    )


def write_notebook(filename, title, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"name": title, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print("wrote", path)


SETUP_CODE_SPARSE = '''\
# --- Setup -------------------------------------------------------------
# Makes the project importable whether this notebook runs standalone in
# Google Colab or from a local checkout, then imports the shared libraries
# used throughout. Safe to re-run.
import os
import sys
import subprocess

REPO_URL = "{repo_url}"


def _locate_repo():
    for candidate in (".", "..", "../..", "repo"):
        path = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(path, "hmb_kuramoto_ode")):
            return path
    # A sparse, blobless clone: this repository also bundles a ~490 MB STEW
    # dataset/ folder (see notebook 07), which this notebook does not need.
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none",
         "--no-checkout", "--sparse", REPO_URL, "repo"],
        check=True,
    )
    subprocess.run(["git", "-C", "repo", "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(
        ["git", "-C", "repo", "sparse-checkout", "set", "hmb_kuramoto_ode", "examples", "configs", "pyproject.toml"],
        check=True,
    )
    subprocess.run(["git", "-C", "repo", "checkout", "--quiet"], check=True)
    return os.path.abspath("repo")


REPO_DIR = _locate_repo()
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch>=2.5,<3", "numpy>=2,<3", "scipy>=1.14,<2", "scikit-learn>=1.5,<2", "matplotlib>=3.9,<4"],
    check=True,
)

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from hmb_kuramoto_ode.utils.seed import seed_everything

print(f"repository: {{REPO_DIR}}")
print(f"torch {{torch.__version__}} | CUDA available: {{torch.cuda.is_available()}}")
'''.format(repo_url=REPO_URL)


# ======================================================================
# Notebook 01 - message passing
# ======================================================================
NB01_INTRO = f"""# 01 · Message passing on the hierarchical rhythm graph

{badge("01_message_passing.ipynb")}

Every model in this repository — from the plain GAT baseline to the full
Kuramoto-attention Neural ODE — is built on the same primitive: **message
passing**, where each node updates its state from its graph neighbors'
states. This notebook makes that primitive concrete on the project's real
graph: 14 EEG electrodes (outer level), each containing 5 rhythm nodes —
delta, theta, alpha, beta, gamma (inner level) — connected by
*same-band spatial* edges (same rhythm, neighboring electrodes) and
*cross-frequency local* edges (different rhythms, same electrode).

No trained model is needed to see the key idea: **information can only
reach a node through the edges that touch it, one hop at a time.** Later
notebooks replace the plain averaging used here with learned attention
(02), diffusion (03), phase synchronization (04), and finally a continuous
version of all of it integrated by an ODE solver (05, 06)."""

NB01_C1 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, SPATIAL_SAME_BAND, CROSS_FREQUENCY_LOCAL
from hmb_kuramoto_ode.contracts import BANDS

REGIONS = 4  # a small number of electrodes keeps the plots legible; the real STEW graph uses 14
edge_index, edge_types = build_hierarchical_graph(REGIONS)
num_nodes = REGIONS * 5

print(f"{num_nodes} nodes ({REGIONS} electrodes x 5 rhythms: {', '.join(BANDS)})")
print(f"{edge_index.shape[1]} directed edges")
print(f"  same-band spatial edges:     {(edge_types == SPATIAL_SAME_BAND).sum().item()}")
print(f"  cross-frequency local edges: {(edge_types == CROSS_FREQUENCY_LOCAL).sum().item()}")
"""

NB01_M2 = """## 1. Visualizing the graph

Each column below is one electrode; each row is one rhythm. Blue edges connect
the *same* rhythm across neighboring electrodes; orange edges connect
*different* rhythms within one electrode."""

NB01_C2 = """fig, ax = plt.subplots(figsize=(7, 4))
positions = {n: (n // 5, n % 5) for n in range(num_nodes)}
src, dst = edge_index
colors = {int(SPATIAL_SAME_BAND): "#4C72B0", int(CROSS_FREQUENCY_LOCAL): "#DD8452"}
seen = set()
for s, d, t in zip(src.tolist(), dst.tolist(), edge_types.tolist()):
    if (d, s) in seen:  # each undirected pair is stored as two directed edges
        continue
    seen.add((s, d))
    (x0, y0), (x1, y1) = positions[s], positions[d]
    label = "same-band spatial" if t == SPATIAL_SAME_BAND else "cross-frequency local"
    already = label in ax.get_legend_handles_labels()[1]
    ax.plot([x0, x1], [y0, y1], color=colors[t], alpha=0.6, zorder=1, label=None if already else label)
for n, (x, y) in positions.items():
    ax.scatter(x, y, s=280, color="white", edgecolor="black", zorder=2)
    ax.text(x, y, BANDS[y][0].upper(), ha="center", va="center", fontsize=9, zorder=3)
ax.set_xticks(range(REGIONS)); ax.set_xticklabels([f"electrode {i}" for i in range(REGIONS)])
ax.set_yticks(range(5)); ax.set_yticklabels(BANDS)
ax.set_title("Hierarchical rhythm graph (letter = band initial)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()
"""

NB01_M3 = """## 2. One step of message passing

The simplest possible aggregation: a node's new state is the **mean of its
incoming neighbors' states**. No learned weights yet — this is the primitive
that `EdgeAttention` (notebook 02) and the ODE vector field (notebooks 03-06)
replace with something learned."""

NB01_C3 = """def mean_aggregate(h, edge_index, num_nodes):
    src, dst = edge_index
    out = torch.zeros_like(h)
    out.index_add_(0, dst, h[src])
    degree = torch.zeros(num_nodes, dtype=h.dtype).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    return out / degree.clamp(min=1)[:, None]
"""

NB01_M4 = """## 3. Watching a signal spread

Place a unit impulse on a single node (electrode 0, delta band) and apply
`mean_aggregate` repeatedly. Because the graph only has direct edges to
same-band neighbors and same-electrode rhythms, the signal needs multiple
hops to reach a node that is both spatially and spectrally far away —
exactly the "receptive field grows by depth" limitation that motivates
continuous, always-on coupling in the full model."""

NB01_C4 = """seed_everything(7)
signal = torch.zeros(num_nodes, 1)
source_node = 0  # electrode 0, delta band
signal[source_node] = 1.0

HOPS = 6
history = [signal.squeeze(1).clone()]
h = signal
for _ in range(HOPS):
    h = mean_aggregate(h, edge_index, num_nodes)
    history.append(h.squeeze(1).clone())
history = torch.stack(history)  # [hops+1, num_nodes]

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(history.T, aspect="auto", cmap="viridis")
ax.set_xlabel("message-passing hop"); ax.set_ylabel("node index")
ax.set_title(f"Signal spreading from node {source_node} (electrode 0, delta)")
fig.colorbar(im, ax=ax, label="propagated magnitude")
plt.show()

far_node = num_nodes - 1  # last electrode, gamma band: spatially and spectrally farthest
print(f"magnitude at the farthest node ({far_node}) per hop:")
for hop, value in enumerate(history[:, far_node].tolist()):
    print(f"  hop {hop}: {value:.4f}")
"""

NB01_M5 = """## Key takeaway

Plain message passing only mixes information along explicit edges, one hop
per layer/step. The rest of this repository keeps that graph structure but
changes *how* neighbors are weighted (attention, notebook 02), *what* gets
mixed (diffusion vs. phase-only coupling, notebooks 03-04), and *how long*
mixing runs — a fixed number of layers vs. a continuously integrated ODE
(notebooks 05-06)."""

NB01_CELLS = [M(NB01_INTRO), C(SETUP_CODE_SPARSE), C(NB01_C1), M(NB01_M2), C(NB01_C2),
              M(NB01_M3), C(NB01_C3), M(NB01_M4), C(NB01_C4), M(NB01_M5)]


# ======================================================================
# Notebook 02 - GAT attention
# ======================================================================
NB02_INTRO = f"""# 02 · GAT-style edge attention

{badge("02_gat_attention.ipynb")}

This notebook looks at `hmb_kuramoto_ode.models.attention.EdgeAttention`,
the learned-weighting ingredient used both by the GAT baseline and inside
the full model's continuous vector field. For every directed edge `(src,
dst)` it scores `[h_src, h_dst, edge_weight]` with a small MLP and then
applies a **segment softmax**: scores are normalized separately for each
destination node, so the incoming edge weights of any node always sum to 1
— exactly like standard graph attention (GAT), restricted to a node's
actual graph neighbors rather than the whole graph."""

NB02_C1 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, SPATIAL_SAME_BAND, CROSS_FREQUENCY_LOCAL
from hmb_kuramoto_ode.models.attention import EdgeAttention

seed_everything(7)
REGIONS = 4
edge_index, edge_types = build_hierarchical_graph(REGIONS)
num_nodes = REGIONS * 5
hidden = 8

attention = EdgeAttention(hidden)
h = torch.randn(num_nodes, hidden)
alpha = attention(h, edge_index)
print(f"{alpha.shape[0]} edge weights, one per directed edge")

# Segment softmax: weights of edges pointing INTO the same destination node sum to 1.
dst = edge_index[1]
incoming_sums = torch.zeros(num_nodes).index_add_(0, dst, alpha)
print("max deviation of per-node incoming weights from 1.0:", (incoming_sums - 1).abs().max().item())
"""

NB02_M2 = """## Training attention on a toy classification task

To see attention specialize, we train a tiny two-layer edge-attention
classifier (the same building block as `GATClassifier` in
`examples/stew_real_experiment.py`, just smaller) on a synthetic task: two
classes of graph samples, where class 1 carries a stronger, alpha-band
signal. If attention is doing its job, the edges touching the alpha-band
nodes should end up weighted differently from the rest once training
finishes."""

NB02_C2 = """class TinyGAT(nn.Module):
    \"\"\"Two-layer edge-attention network over flattened rhythm nodes.\"\"\"
    def __init__(self, edge_index, features, hidden=16, classes=2):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.input = nn.Linear(features, hidden)
        self.attn1 = EdgeAttention(hidden)
        self.proj1 = nn.Linear(hidden, hidden)
        self.attn2 = EdgeAttention(hidden)
        self.proj2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, classes)

    def layer(self, h, attn, proj):
        src, dst = self.edge_index
        alpha = attn(h, self.edge_index)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, alpha[:, None] * proj(h[src]))
        return torch.relu(out + h)

    def forward(self, x):
        batch, nodes, _ = x.shape
        h = torch.relu(self.input(x.reshape(-1, x.shape[-1])))
        h = self.layer(h, self.attn1, self.proj1)
        h = self.layer(h, self.attn2, self.proj2)
        return self.head(h.reshape(batch, nodes, -1).mean(1))
"""

NB02_C3 = """FEATURES = 6  # mirrors the 6 real rhythm features (log/relative power, amplitude, phase sin/cos, entropy)
ALPHA_BAND_NODES = torch.arange(2, num_nodes, 5)  # node index 2, 7, 12, ... is the alpha rhythm of each electrode


def synthetic_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, num_nodes, FEATURES, generator=g) * 0.5
    x[:, ALPHA_BAND_NODES, :] += labels.float()[:, None, None] * 1.5
    return x, labels


xtr, ytr = synthetic_batch(128, seed=1)
xval, yval = synthetic_batch(32, seed=2)

model = TinyGAT(edge_index, FEATURES, hidden=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


def mean_attention_by_type(model, x):
    with torch.no_grad():
        h = torch.relu(model.input(x.reshape(-1, x.shape[-1])))
        alpha = model.attn1(h, model.edge_index)
    return {
        "same-band spatial": float(alpha[edge_types == SPATIAL_SAME_BAND].mean()),
        "cross-frequency local": float(alpha[edge_types == CROSS_FREQUENCY_LOCAL].mean()),
    }


before = mean_attention_by_type(model, xtr)
history = []
for epoch in range(60):
    model.train()
    optimizer.zero_grad()
    logits = model(xtr)
    loss = nn.functional.cross_entropy(logits, ytr)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        val_acc = (model(xval).argmax(1) == yval).float().mean()
    history.append({"epoch": epoch, "loss": loss.item(), "val_accuracy": float(val_acc)})
after = mean_attention_by_type(model, xtr)

print("mean attention weight by edge type")
print(f"  before training: {before}")
print(f"  after training:  {after}")
"""

NB02_C4 = """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = [h["epoch"] for h in history]
axes[0].plot(epochs, [h["loss"] for h in history])
axes[0].set(xlabel="epoch", ylabel="training loss", title="TinyGAT training loss")
axes[1].plot(epochs, [h["val_accuracy"] for h in history], color="tab:green")
axes[1].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
labels = list(before.keys())
xpos = np.arange(len(labels))
ax.bar(xpos - 0.18, [before[l] for l in labels], width=0.36, label="before training")
ax.bar(xpos + 0.18, [after[l] for l in labels], width=0.36, label="after training")
ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("mean attention weight")
ax.set_title("Does attention specialize by edge type?")
ax.legend()
fig.tight_layout()
plt.show()
"""

NB02_M3 = """## Key takeaway

Segment softmax gives every node a proper probability distribution over its
own neighbors, and — unlike the fixed mean aggregation in notebook 01 — that
distribution is *learned* to fit the task. The same `EdgeAttention` module
reappears unchanged inside `KuramotoGrandVectorField` (notebooks 03-06),
where it weights both the diffusion term and the phase-coupling term."""

NB02_CELLS = [M(NB02_INTRO), C(SETUP_CODE_SPARSE), C(NB02_C1), M(NB02_M2), C(NB02_C2), C(NB02_C3), C(NB02_C4), M(NB02_M3)]


# ======================================================================
# Notebook 03 - GRAND diffusion
# ======================================================================
NB03_INTRO = f"""# 03 · GRAND-style graph diffusion

{badge("03_grand_diffusion.ipynb")}

GRAND (GRAph Neural Diffusion) treats a graph neural network as the discretization
of a diffusion PDE: every node's *entire* feature vector is pulled toward an
attention-weighted average of its neighbors. This notebook reproduces exactly
the diffusion term used inside `KuramotoGrandVectorField.forward` —

```python
grand.index_add_(0, dst, alpha[:, None] * (h[src] - h[dst]))
```

— outside the full model, so the smoothing behavior is visible on its own
before notebook 06 combines it with Kuramoto coupling through a learned gate."""

NB03_C1 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph
from hmb_kuramoto_ode.models.attention import EdgeAttention

seed_everything(7)
REGIONS = 4
edge_index, edge_types = build_hierarchical_graph(REGIONS)
num_nodes = REGIONS * 5
hidden = 3

# Give each electrode's five rhythm nodes a different baseline level, plus
# noise, to mimic real inter-electrode amplitude differences before any
# mixing happens.
electrode_level = torch.randn(REGIONS, 1, hidden) * 2.0
h0 = (electrode_level.expand(REGIONS, 5, hidden) + 0.3 * torch.randn(REGIONS, 5, hidden)).reshape(num_nodes, hidden)
print("initial per-node feature variance:", float(h0.var()))
"""

NB03_M2 = """## Diffusion step

Same-attention-weighted difference between a node and its neighbors, scaled
by a small step `dt` — a literal copy of the model's GRAND branch, using a
fresh (untrained) `EdgeAttention` module, exactly as `KuramotoGrandVectorField`
does before any training happens."""

NB03_C2 = """attention = EdgeAttention(hidden)  # same module class used inside the real vector field


def grand_step(h, edge_index, attention, dt=0.2):
    src, dst = edge_index
    alpha = attention(h, edge_index)
    diffusion = torch.zeros_like(h)
    diffusion.index_add_(0, dst, alpha[:, None] * (h[src] - h[dst]))
    return h + dt * diffusion


STEPS = 40
h = h0.clone()
variances = [h.var().item()]
feature0_over_time = [h[:, 0].detach().clone()]
for _ in range(STEPS):
    h = grand_step(h, edge_index, attention)
    variances.append(h.var().item())
    feature0_over_time.append(h[:, 0].detach().clone())
feature0_over_time = torch.stack(feature0_over_time)  # [steps+1, num_nodes]
"""

NB03_C3 = """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(variances)
axes[0].set(xlabel="diffusion step", ylabel="feature variance across nodes",
            title="GRAND smooths the whole feature vector toward consensus")
im = axes[1].imshow(feature0_over_time.T, aspect="auto", cmap="coolwarm")
axes[1].set(xlabel="diffusion step", ylabel="node index", title="feature 0 per node over time")
fig.colorbar(im, ax=axes[1])
fig.tight_layout()
plt.show()

print(f"feature variance: {variances[0]:.4f} -> {variances[-1]:.4f} after {STEPS} diffusion steps")
"""

NB03_M3 = """## Key takeaway

GRAND diffuses **every** latent coordinate toward a local consensus — it
has no notion of "phase" and will happily average an angular coordinate the
same way it averages any other feature, which is exactly why the full
model keeps a *separate* Kuramoto term (notebook 04) for the angular
coordinate and gates between the two (notebook 06) rather than diffusing
everything indiscriminately."""

NB03_CELLS = [M(NB03_INTRO), C(SETUP_CODE_SPARSE), C(NB03_C1), M(NB03_M2), C(NB03_C2), C(NB03_C3), M(NB03_M3)]


# ======================================================================
# Notebook 04 - Kuramoto synchronization
# ======================================================================
NB04_INTRO = f"""# 04 · Kuramoto phase synchronization

{badge("04_kuramoto_sync.ipynb")}

The classic Kuramoto model couples oscillator *phases* through a sine of
their pairwise difference:

```
d(theta_i)/dt = omega_i + K * mean_j sin(theta_j - theta_i)
```

Synchronization is measured by the **order parameter**
`r = |mean_j exp(i * theta_j)|`, which goes from 0 (phases spread evenly
around the circle) to 1 (all phases equal). This is exactly the mechanism
behind latent coordinate 0 in the full model — see
`KuramotoGrandVectorField.forward`'s `kur[:, 0]` term — restricted here to
a small subgraph so the dynamics are easy to see on their own."""

NB04_C1 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, SPATIAL_SAME_BAND

REGIONS = 8
edge_index, edge_types = build_hierarchical_graph(REGIONS)
ALPHA_BAND = 2  # delta=0, theta=1, alpha=2, beta=3, gamma=4

# Keep only the same-band spatial edges of the alpha rhythm, and collapse
# "electrode*5 + band" node indices down to one index per electrode: this
# reproduces the real same-band spatial coupling structure (electrodes in a
# chain) restricted to a single rhythm, giving a clean set of oscillators.
mask = (edge_types == SPATIAL_SAME_BAND) & (edge_index[0] % 5 == ALPHA_BAND)
kuramoto_edges = edge_index[:, mask] // 5
n = REGIONS
print(f"{n} coupled alpha-band oscillators (one per electrode), {kuramoto_edges.shape[1]} directed couplings")
"""

NB04_C2 = """def order_parameter(phase):
    z = torch.complex(torch.cos(phase), torch.sin(phase))
    return z.mean().abs().item()


def kuramoto_step(phase, edges, omega, K, dt=0.05):
    src, dst = edges
    coupling = torch.zeros_like(phase)
    coupling.index_add_(0, dst, torch.sin(phase[src] - phase[dst]))
    degree = torch.zeros_like(phase).index_add_(0, dst, torch.ones_like(dst, dtype=phase.dtype))
    coupling = coupling / degree.clamp(min=1)
    return phase + dt * (omega + K * coupling)
"""

NB04_M2 = """## Coupling strength controls synchronization

Below the same natural frequencies `omega` are simulated under four
coupling strengths `K`. Weak coupling never synchronizes; strong coupling
locks all oscillators to a common phase — the same phase transition studied
in the original Kuramoto model."""

NB04_C3 = """seed_everything(7)
omega = torch.randn(n) * 0.5
STEPS = 400
results = {}
for K in (0.0, 0.5, 2.0, 6.0):
    phase = torch.rand(n) * 2 * torch.pi
    trace = [order_parameter(phase)]
    for _ in range(STEPS):
        phase = kuramoto_step(phase, kuramoto_edges, omega, K)
        trace.append(order_parameter(phase))
    results[K] = trace

fig, ax = plt.subplots(figsize=(7, 4))
for K, trace in results.items():
    ax.plot(trace, label=f"K={K}")
ax.set(xlabel="integration step", ylabel="order parameter r(t)",
       title="Coupling strength controls synchronization")
ax.legend()
fig.tight_layout()
plt.show()
"""

NB04_C4 = """K = 6.0
seed_everything(11)
phase = torch.rand(n) * 2 * torch.pi
snap_at = {0: "start", 40: "mid", 400: "end"}
snapshots = {snap_at[0]: phase.clone()}
for step in range(1, 401):
    phase = kuramoto_step(phase, kuramoto_edges, omega, K=K)
    if step in snap_at:
        snapshots[snap_at[step]] = phase.clone()

fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "polar"})
for ax, (name, ph) in zip(axes, snapshots.items()):
    ax.scatter(ph.numpy(), np.ones(n), s=80)
    ax.set_title(name)
    ax.set_yticklabels([])
fig.suptitle(f"Phases converge onto the unit circle as they synchronize (K={K})")
fig.tight_layout()
plt.show()
"""

NB04_M3 = """## Bridging to the project's differentiable solver

The manual Euler loop above is a teaching simplification. The real model
integrates its dynamics with `hmb_kuramoto_ode.models.ode_solver.integrate`,
which supports both Euler and a differentiable RK4 and keeps every
intermediate state on the autograd graph. Wrapping the same coupling rule as
a `field(t, phase)` function lets us run it through that exact solver."""

NB04_C5 = """from hmb_kuramoto_ode.models.ode_solver import integrate


def kuramoto_field(t, phase):
    src, dst = kuramoto_edges
    coupling = torch.zeros_like(phase)
    coupling.index_add_(0, dst, torch.sin(phase[src] - phase[dst]))
    degree = torch.zeros_like(phase).index_add_(0, dst, torch.ones_like(dst, dtype=phase.dtype))
    return omega + K * coupling / degree.clamp(min=1)


seed_everything(11)
phase0 = torch.rand(n) * 2 * torch.pi
final_phase, trajectory = integrate(kuramoto_field, phase0, t1=20.0, step_size=0.05, method="rk4", return_trajectory=True)
r_trace = [order_parameter(p) for p in trajectory]
print(f"order parameter after RK4 integration: {r_trace[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(r_trace)
ax.set(xlabel="RK4 step", ylabel="order parameter",
       title="Same dynamics through the project's differentiable RK4 solver")
fig.tight_layout()
plt.show()
"""

NB04_M4 = """## Key takeaway

Kuramoto coupling only ever touches the angular coordinate, and only
through a sine of the pairwise phase difference — never a linear average.
That is precisely why the full model keeps it separate from the GRAND
diffusion of notebook 03, and combines the two with a learned gate rather
than one generic mixing rule (notebook 06)."""

NB04_CELLS = [M(NB04_INTRO), C(SETUP_CODE_SPARSE), C(NB04_C1), C(NB04_C2), M(NB04_M2), C(NB04_C3), C(NB04_C4),
              M(NB04_M3), C(NB04_C5), M(NB04_M4)]


# ======================================================================
# Notebook 05 - Neural ODE
# ======================================================================
NB05_INTRO = f"""# 05 · Neural ODE integration

{badge("05_neural_ode.ipynb")}

A Neural ODE replaces a stack of discrete layers with a single learned
derivative `dz/dt = f(t, z)`, integrated by a numerical solver. This
repository's solver, `hmb_kuramoto_ode.models.ode_solver.integrate`, is a
fixed-step, fully differentiable Euler/RK4 implementation: every
intermediate stage stays on the autograd graph, so gradients flow through
the *entire* integration, not just the final state. This notebook checks
that claim quantitatively, fits a parameter through the solver, and finally
runs the project's real vector field through it."""

NB05_C1 = """from hmb_kuramoto_ode.models.ode_solver import integrate

a_true = -0.8
def linear_field(t, y):
    return a_true * y

t1 = 1.0
analytic = torch.exp(torch.tensor(a_true * t1))

# Stop at step=0.0625: smaller RK4 steps push the error below float32's
# ~1e-7 relative precision floor, where rounding noise (not truncation
# error) dominates and the 4th-order trend below stops holding.
step_sizes = [0.5, 0.25, 0.125, 0.0625]
errors = {"euler": [], "rk4": []}
for step in step_sizes:
    for method in errors:
        y = integrate(linear_field, torch.tensor(1.0), t1=t1, step_size=step, method=method)
        errors[method].append(abs(float(y) - float(analytic)))

fig, ax = plt.subplots(figsize=(6, 4.5))
for method, values in errors.items():
    ax.loglog(step_sizes, values, marker="o", label=method)
ax.set(xlabel="step size", ylabel="|numerical - analytic|",
       title="RK4 converges to 4th order, Euler to 1st order")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

for method, values in errors.items():
    ratio = values[-2] / values[-1]
    print(f"{method}: halving the step size divides the error by about {ratio:.1f}x")
"""

NB05_M2 = """## Backpropagating through the solver

Now treat the ODE's rate as a learnable `nn.Parameter` and fit it so that
`y(1)` matches a target value, purely by backpropagating through every RK4
stage of `integrate`. This is the same mechanism the full model uses to
learn frequencies, coupling strengths, and gates — the ODE is part of the
computational graph, not a black box."""

NB05_C2 = """seed_everything(7)
target = torch.tensor(0.35)
rate = nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([rate], lr=0.1)


def field(t, y):
    return rate * y


loss_history = []
for step in range(120):
    optimizer.zero_grad()
    y1 = integrate(field, torch.tensor(1.0), t1=1.0, step_size=0.05, method="rk4")
    loss = (y1 - target) ** 2
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(loss_history)
ax.set(xlabel="gradient step", ylabel="(y(1) - target)^2", title="Backpropagating through every RK4 stage")
fig.tight_layout()
plt.show()

fitted_a = rate.item()
true_a = float(torch.log(target))
print(f"fitted rate:      {fitted_a:.4f}")
print(f"analytic optimum: {true_a:.4f}  (= ln(target), since y(1) = exp(rate))")
"""

NB05_M3 = """## The real vector field, integrated

Finally, run `KuramotoGrandVectorField` — the actual Kuramoto+GRAND+attention
dynamics used by the full model — through the same solver on a small
hierarchical graph, and track the angular coordinate (latent dim 0) of every
node across the integration."""

NB05_C3 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph
from hmb_kuramoto_ode.models.ode_func import KuramotoGrandVectorField

seed_everything(3)
REGIONS = 4
edge_index, _ = build_hierarchical_graph(REGIONS)
num_nodes = REGIONS * 5
hidden = 6

field = KuramotoGrandVectorField(hidden, edge_index)
z0 = torch.randn(num_nodes, hidden) * 0.3
z0[:, 0] = torch.rand(num_nodes) * 2 * torch.pi  # angular coordinate, as the encoder would set it

field.nfe = 0
_, trajectory = integrate(field, z0, t1=0.2, step_size=0.02, method="rk4", return_trajectory=True)
print(f"RK4 over {trajectory.shape[0] - 1} steps used {field.nfe} function evaluations (4 per step)")

fig, ax = plt.subplots(figsize=(7, 4))
phase_trajectory = trajectory[:, :, 0].detach().numpy()  # angular coordinate over time, all nodes
ax.plot(phase_trajectory)
ax.set(xlabel="RK4 step", ylabel="phase (latent coordinate 0)", title="Every node's phase during ODE integration")
fig.tight_layout()
plt.show()
"""

NB05_M4 = """## Key takeaway

The solver counts vector-field evaluations (`nfe`) and keeps every one of
them differentiable — RK4 costs 4 evaluations per step for a 4th-order
accuracy gain, and that whole chain backpropagates. This is what "the ODE
is part of the computational graph" means concretely, and it's the same
`integrate()` call used inside `HierarchicalKuramotoODE.forward` in
notebook 06."""

NB05_CELLS = [M(NB05_INTRO), C(SETUP_CODE_SPARSE), C(NB05_C1), M(NB05_M2), C(NB05_C2), M(NB05_M3), C(NB05_C3), M(NB05_M4)]

print("Part 1 of the generator module loaded (notebooks 01-05 defined).")

# ======================================================================
# Notebook 06 - synthetic multitask full model
# ======================================================================
NB06_INTRO = f"""# 06 · Training the full multitask model on synthetic data

{badge("06_synthetic_multitask.ipynb")}

`HierarchicalKuramotoODE` combines everything from notebooks 01-05: the
biomedical encoder, the Kuramoto+GRAND+attention vector field integrated by
the differentiable RK4 solver, hierarchical attention pooling, and three
task heads — graph classification, node-feature reconstruction, and link
prediction. This notebook trains it on a synthetic dataset (real STEW
training is notebook 07) so the mechanics are visible without needing the
dataset, and closes with an ablation comparison of the Kuramoto and GRAND
switches described in the project README."""

NB06_C1 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, batch_edges
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE
from hmb_kuramoto_ode.contracts import BANDS
from sklearn.metrics import roc_auc_score

REGIONS = 4
FEATURES = 6
ALPHA_BAND = 2
base_edges, edge_types = build_hierarchical_graph(REGIONS)
nodes_per_graph = REGIONS * 5


def synthetic_dataset(n, seed):
    \"\"\"Class 1 gets stronger alpha-band power and a phase shift; everything else is noise.\"\"\"
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, REGIONS, 5, FEATURES, generator=g) * 0.4
    boost = labels.float()[:, None]  # [n, 1], broadcasts against the [n, regions] band slices below
    x[:, :, ALPHA_BAND, 0] += boost * 1.2                      # log power
    x[:, :, ALPHA_BAND, 1] += boost * 0.6                      # relative power
    phase_shift = boost * (torch.pi / 2)
    x[:, :, ALPHA_BAND, 2] = torch.sin(phase_shift)            # overwrite sin(phase)
    x[:, :, ALPHA_BAND, 3] = torch.cos(phase_shift)            # overwrite cos(phase)
    return x, labels


TRAIN_N, VAL_N = 64, 16
xtr, ytr = synthetic_dataset(TRAIN_N, seed=1)
xval, yval = synthetic_dataset(VAL_N, seed=2)
print(f"train {tuple(xtr.shape)}, validation {tuple(xval.shape)}")
"""

NB06_M2 = """## Link-prediction pairs

For each graph in the batch we sample a handful of real edges (positives)
and random node pairs (negatives, occasionally a real edge by chance — an
acceptable simplification for this toy demo; real STEW training instead
uses the canonical, leakage-checked negative sampling in `data/splits.py`)."""

NB06_C2 = """def sample_link_pairs(batch_size, nodes_per_graph, base_edges, k_pos=6, k_neg=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    positives, negatives = [], []
    for b in range(batch_size):
        offset = b * nodes_per_graph
        perm = torch.randperm(base_edges.shape[1], generator=g)[:k_pos]
        positives.append(base_edges[:, perm] + offset)
        neg_u = torch.randint(0, nodes_per_graph, (k_neg,), generator=g) + offset
        neg_v = torch.randint(0, nodes_per_graph, (k_neg,), generator=g) + offset
        negatives.append(torch.stack([neg_u, neg_v]))
    pos, neg = torch.cat(positives, dim=1), torch.cat(negatives, dim=1)
    pairs = torch.cat([pos, neg], dim=1)
    targets = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    return pairs, targets
"""

NB06_M3 = """## Training loop

The combined loss adds graph cross-entropy, node-reconstruction MSE
(reconstructing all nodes here, rather than the masked subset the real
training loop uses, for clarity), and link binary cross-entropy — exactly
the three heads returned by `HierarchicalKuramotoODE.forward`. As in
`examples/stew_real_experiment.py`, the model's edge buffer is swapped
between the train- and validation-sized batches around each evaluation."""

NB06_C3 = """train_edges = batch_edges(base_edges, nodes_per_graph, TRAIN_N)
val_edges = batch_edges(base_edges, nodes_per_graph, VAL_N)


def build_model(**ablations):
    seed_everything(0)
    return HierarchicalKuramotoODE(train_edges, features=FEATURES, hidden=16, **ablations)


def train_model(model, epochs=40, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        model.field.edge_index = train_edges
        optimizer.zero_grad()
        pairs, link_targets = sample_link_pairs(TRAIN_N, nodes_per_graph, base_edges, seed=epoch)
        out = model(xtr, edge_pairs=pairs)
        graph_loss = nn.functional.cross_entropy(out["graph_logits"], ytr)
        node_loss = out["node_prediction"].sub(xtr).square().mean()
        link_loss = nn.functional.binary_cross_entropy_with_logits(out["link_logits"], link_targets)
        loss = graph_loss + 0.3 * node_loss + 0.3 * link_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        model.field.edge_index = val_edges
        with torch.no_grad():
            val_pairs, val_link_targets = sample_link_pairs(VAL_N, nodes_per_graph, base_edges, seed=1000 + epoch)
            val_out = model(xval, edge_pairs=val_pairs)
            val_graph_acc = (val_out["graph_logits"].argmax(1) == yval).float().mean().item()
            val_link_prob = torch.sigmoid(val_out["link_logits"]).numpy()
            val_link_auc = roc_auc_score(val_link_targets.numpy(), val_link_prob)
            val_node_mse = val_out["node_prediction"].sub(xval).square().mean().item()
        history.append(dict(epoch=epoch, loss=loss.item(), graph_loss=graph_loss.item(), node_loss=node_loss.item(),
                             link_loss=link_loss.item(), val_graph_acc=val_graph_acc, val_link_auc=val_link_auc,
                             val_node_mse=val_node_mse))
        model.field.edge_index = train_edges
    return history, model


history, model = train_model(build_model(), epochs=40)
print(f"final validation graph accuracy: {history[-1]['val_graph_acc']:.3f}")
print(f"final validation link AUC:       {history[-1]['val_link_auc']:.3f}")
"""

NB06_C4 = """fig, axes = plt.subplots(2, 2, figsize=(11, 8))
epochs = [h["epoch"] for h in history]
axes[0, 0].plot(epochs, [h["loss"] for h in history], label="total", linewidth=2)
axes[0, 0].plot(epochs, [h["graph_loss"] for h in history], label="graph CE", alpha=0.7)
axes[0, 0].plot(epochs, [h["node_loss"] for h in history], label="node MSE", alpha=0.7)
axes[0, 0].plot(epochs, [h["link_loss"] for h in history], label="link BCE", alpha=0.7)
axes[0, 0].set_title("Training loss"); axes[0, 0].legend(fontsize=8)
axes[0, 1].plot(epochs, [h["val_graph_acc"] for h in history], color="tab:green")
axes[0, 1].set_title("Validation graph-classification accuracy"); axes[0, 1].set_ylim(0, 1.02)
axes[1, 0].plot(epochs, [h["val_link_auc"] for h in history], color="tab:purple")
axes[1, 0].set_title("Validation link-prediction ROC-AUC"); axes[1, 0].set_ylim(0, 1.02)
axes[1, 1].plot(epochs, [h["val_node_mse"] for h in history], color="tab:red")
axes[1, 1].set_title("Validation node-reconstruction MSE")
for ax in axes.flat:
    ax.set_xlabel("epoch")
fig.tight_layout()
plt.show()
"""

NB06_M4 = """## What did pooling learn to trust?

The synthetic signal was injected into the alpha band only. `out["attention"]["rhythm"]`
is the hierarchical pooling weight each rhythm receives when it's summarized
into its electrode — a direct, visual check of whether the model learned to
lean on the band that actually carries the label."""

NB06_C5 = """model.eval()
model.field.edge_index = val_edges
with torch.no_grad():
    val_pairs, _ = sample_link_pairs(VAL_N, nodes_per_graph, base_edges, seed=99)
    out = model(xval, edge_pairs=val_pairs)

rhythm_attention = out["attention"]["rhythm"].mean(dim=(0, 1)).detach().numpy()  # mean over batch & electrode -> [5]
fig, ax = plt.subplots(figsize=(5, 4))
colors = ["#DD8452" if band == BANDS[ALPHA_BAND] else "#4C72B0" for band in BANDS]
ax.bar(BANDS, rhythm_attention, color=colors)
ax.set_ylabel("mean pooling attention")
ax.set_title(f"Pooling weight per rhythm (signal was injected in {BANDS[ALPHA_BAND]})")
fig.tight_layout()
plt.show()
"""

NB06_M5 = """## Ablating Kuramoto and GRAND

`HierarchicalKuramotoODE` accepts the same ablation switches as the CLI
(`kuramoto=False`, `grand=False`, `cross_frequency=False`, `residual=False`).
Below, four short training runs compare the full model against disabling
each coupling mechanism, holding everything else fixed. On a task this
small, every variant eventually reaches similar *final* accuracy — the
encoder and pooling head alone can already fit it — so the more honest
comparison is the *learning curve*: how quickly each variant gets there,
which is where the coupling mechanisms actually earn their keep."""

NB06_C6 = """ablation_settings = {
    "full (Kuramoto+GRAND)": {},
    "Kuramoto-only": {"grand": False},
    "GRAND-only": {"kuramoto": False},
    "neither (residual only)": {"grand": False, "kuramoto": False},
}
ablation_curves = {}
for name, ablation in ablation_settings.items():
    ablation_history, _ = train_model(build_model(**ablation), epochs=15)
    ablation_curves[name] = [h["val_graph_acc"] for h in ablation_history]
    print(f"{name}: validation accuracy after 15 epochs = {ablation_curves[name][-1]:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for name, curve in ablation_curves.items():
    ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=3, label=name)
ax.set(xlabel="epoch", ylabel="validation graph accuracy", title="Ablating the two coupling mechanisms")
ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

NB06_M6 = """## Key takeaway

This is a small, synthetic, single-seed comparison — not a claim about
which mechanism matters more on real EEG (see notebook 07 and the README's
"no superiority claimed" note), and the curves above can reorder with a
different seed or a slightly larger/smaller signal. What it does show
concretely is that the ablation switches actually change the model's
learning dynamics, and that the combined loss, the pooling attention, and
the link/node/graph heads all train together through one shared ODE
integration, as described in the project README's "How the code works"
section."""

NB06_CELLS = [M(NB06_INTRO), C(SETUP_CODE_SPARSE), C(NB06_C1), M(NB06_M2), C(NB06_C2), M(NB06_M3), C(NB06_C3), C(NB06_C4),
              M(NB06_M4), C(NB06_C5), M(NB06_M5), C(NB06_C6), M(NB06_M6)]

print("Part 2 of the generator module loaded (notebook 06 defined).")

# ======================================================================
# Notebook 07 - real STEW experiment
# ======================================================================
SETUP_CODE_FULL = '''\
# --- Setup -------------------------------------------------------------
# Makes the project importable whether this notebook runs standalone in
# Google Colab or from a local checkout, then imports the shared libraries
# used throughout. Safe to re-run.
import os
import sys
import subprocess

REPO_URL = "{repo_url}"


def _locate_repo():
    for candidate in (".", "..", "../..", "repo"):
        path = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(path, "hmb_kuramoto_ode")):
            return path
    # Unlike the other example notebooks, this one needs the STEW recordings,
    # so it clones the full repository history -- including the ~490 MB
    # dataset/ folder bundled in this project (see the README's "STEW dataset
    # and ratings.txt" section). That can take a minute or two on Colab's
    # network. If you already have your own STEW copy (e.g. on Google Drive),
    # skip this cell and set DATA_ROOT to that path instead.
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", REPO_URL, "repo"], check=True)
    return os.path.abspath("repo")


REPO_DIR = _locate_repo()
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch>=2.5,<3", "numpy>=2,<3", "scipy>=1.14,<2", "scikit-learn>=1.5,<2",
     "PyYAML>=6,<7", "matplotlib>=3.9,<4", "tqdm"],
    check=True,
)

DATA_ROOT = os.path.join(REPO_DIR, "dataset")
assert os.path.isdir(DATA_ROOT), f"expected STEW recordings under {{DATA_ROOT}}"

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from hmb_kuramoto_ode.utils.seed import seed_everything

print(f"repository:   {{REPO_DIR}}")
print(f"STEW data root: {{DATA_ROOT}}")
print(f"torch {{torch.__version__}} | CUDA available: {{torch.cuda.is_available()}}")
'''.format(repo_url=REPO_URL)

NB07_INTRO = f"""# 07 · Training on real STEW recordings

{badge("07_stew_real_experiment.ipynb")}

This notebook is a Colab-friendly walkthrough of
`examples/stew_real_experiment.py`: it loads real STEW EEG windows, keeps
subjects disjoint across train/validation/test, trains both the GAT
baseline and the full Kuramoto-attention Neural ODE, and plots the results.
It never substitutes synthetic data — if `dataset/` is missing or malformed
the loader raises rather than making something up.

**What STEW is.** STEW ("Simultaneous Task EEG Workload", Lim, Sourina &
Wang, 2018) recorded 48 participants on a 14-channel Emotiv EPOC headset at
128 Hz doing two sessions each: a resting/low-workload baseline (`lo`) and
the demanding SIMKAP multitasking test (`hi`). The binary label used
everywhere below comes only from that `hi`/`lo` filename suffix.
`ratings.txt`, also bundled in this repository, holds each participant's own
1-9 subjective workload rating per session (`subject_id, rating_lo,
rating_hi`) — it is deliberately **not** used as a label or model input
here (see `data/stew.py`'s `condition_from_path`), only as optional
reference data for anyone who wants to correlate predictions with
self-reports."""

NB07_M1 = """## Alternative: bring your own STEW copy

If you'd rather not wait for the ~490 MB clone, mount Google Drive first and
point `DATA_ROOT` at your own copy before running the setup cell (STEW files
must be named like `sub01_hi.txt` / `sub01_lo.txt`, 14 numeric EEG columns
each):

```python
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/STEW'
```"""

NB07_M2 = """## 1. A look at one recording

One raw channel, and the six interpretable rhythm features
(`RhythmPreprocessor.transform_window`: log power, relative power, Hilbert
amplitude, phase sine/cosine, spectral entropy) extracted for every band of
the first electrode in the first analysis window."""

NB07_C1 = """from hmb_kuramoto_ode.data.stew import STEWDataset
from hmb_kuramoto_ode.data.preprocessing import RhythmPreprocessor
from hmb_kuramoto_ode.contracts import DEFAULT_CHANNELS, BANDS

preprocessor = RhythmPreprocessor()
dataset = STEWDataset(DATA_ROOT, preprocessor)
print(dataset.inspect())

record = dataset.records[0]
raw = dataset.load(record)  # [14, samples]
print(f"{record.path.name}: subject {record.subject_id}, condition '{record.condition}', shape {raw.shape}")

fig, ax = plt.subplots(figsize=(9, 3))
seconds = np.arange(raw.shape[1]) / preprocessor.sfreq
five_seconds = int(preprocessor.sfreq * 5)
ax.plot(seconds[:five_seconds], raw[0, :five_seconds])
ax.set(xlabel="time (s)", ylabel="amplitude", title=f"{DEFAULT_CHANNELS[0]} channel, first 5 s of {record.path.name}")
fig.tight_layout()
plt.show()

start, window = preprocessor.windows(raw)[0]
features = preprocessor.transform_window(window)  # [14, 5, 6]
feature_names = ["log power", "relative power", "amplitude", "sin(phase)", "cos(phase)", "spectral entropy"]

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(features[0], aspect="auto", cmap="viridis")
ax.set_xticks(range(6)); ax.set_xticklabels(feature_names, rotation=30, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(BANDS)
ax.set_title(f"Six rhythm features per band, electrode {DEFAULT_CHANNELS[0]}")
fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()
"""

NB07_M3 = """## 2. Subject-disjoint windows and split

`examples.stew_real_experiment` provides the same window-loading and
subject-grouped split used by the CLI: the split groups by subject, so no
subject's windows appear in more than one of train/validation/test."""

NB07_C2 = """import random

import examples.stew_real_experiment as real_experiment

seed_everything(7)
rows = real_experiment.load_windows(DATA_ROOT, limit_per_record=4)
print(f"{len(rows)} windows from {len({r[2] for r in rows})} subjects")

(train_rows, val_rows, test_rows), subjects = real_experiment.split_rows(rows)

# Subsample the training windows: with the real 14-electrode STEW graph,
# EdgeAttention's per-destination-node normalization scales steeply with
# batch size, so a small batch keeps this interactive notebook fast on a
# Colab CPU. Drop this cap (or raise it) to train on every window, or switch
# to configs/stew_full.yaml for the full protocol.
MAX_TRAIN_ROWS = 24
if len(train_rows) > MAX_TRAIN_ROWS:
    train_rows = random.Random(7).sample(train_rows, MAX_TRAIN_ROWS)

print(f"train subjects ({len(subjects[0])}, {len(train_rows)} windows kept):", sorted(subjects[0]))
print(f"validation subject: {subjects[1]} ({len(val_rows)} windows)")
print(f"test subject:       {subjects[2]} ({len(test_rows)} windows)")
"""

NB07_C3 = """from hmb_kuramoto_ode.data.preprocessing import TrainNormalizer

normalizer = TrainNormalizer().fit(np.stack([r[0] for r in train_rows]), [r[2] for r in train_rows])


def tensors(rows):
    features = normalizer.transform(np.stack([r[0] for r in rows]))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([r[1] for r in rows], dtype=torch.long)
    return x, y


xtr, ytr = tensors(train_rows)
xval, yval = tensors(val_rows)
xtest, ytest = tensors(test_rows)
regions = xtr.shape[1]
print(f"train {tuple(xtr.shape)}, validation {tuple(xval.shape)}, test {tuple(xtest.shape)}")
"""

NB07_M4 = """## 3. Training GAT and the full model

Both models share the same training loop (mirroring `main()` in
`examples/stew_real_experiment.py`): full-batch gradient descent per epoch,
gradient clipping, and best-validation-accuracy checkpointing before the
final test evaluation. A small number of epochs and windows keeps this fast
enough for an interactive Colab session — for the actual subject-independent
protocol (grouped cross-validation, early stopping, mixed precision), use
`configs/stew_full.yaml` with `hmb_kuramoto_ode.cli cross-validate`, as
described in the README."""

NB07_C4 = """from hmb_kuramoto_ode.data.graph_builder import build_hierarchical_graph, batch_edges
from hmb_kuramoto_ode.models.full_model import HierarchicalKuramotoODE

base_edges, _ = build_hierarchical_graph(regions)


def run_experiment(model_name, epochs=15, lr=0.01):
    seed_everything(7)
    full = model_name == "full"
    train_edges = batch_edges(base_edges, regions * 5, xtr.shape[0])
    val_edges = batch_edges(base_edges, regions * 5, xval.shape[0])
    test_edges = batch_edges(base_edges, regions * 5, xtest.shape[0])

    model = (HierarchicalKuramotoODE(train_edges, hidden=16) if full
             else real_experiment.GATClassifier(train_edges))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def set_edges(edges):
        if full:
            model.field.edge_index = edges
        else:
            model.edge_index = edges

    history, best = [], None
    for epoch in range(epochs):
        model.train(); set_edges(train_edges); optimizer.zero_grad()
        logits = model(xtr)["graph_logits"] if full else model(xtr)
        loss = nn.functional.cross_entropy(logits, ytr)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        set_edges(val_edges)
        val_metrics, _ = real_experiment.evaluate(model, xval, yval, full)
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_accuracy": val_metrics["accuracy"]})
        state = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("edge_index")}
        if best is None or val_metrics["accuracy"] > best[0]:
            best = (val_metrics["accuracy"], state)

    model.load_state_dict(best[1], strict=False)
    set_edges(test_edges)
    test_metrics, test_probability = real_experiment.evaluate(model, xtest, ytest, full)
    return history, test_metrics, test_probability


results = {}
for name in ("gat", "full"):
    hist, test_metrics, probability = run_experiment(name, epochs=15)
    results[name] = {"history": hist, "test_metrics": test_metrics, "probability": probability}
    print(f"{name}: test accuracy={test_metrics['accuracy']:.3f} roc_auc={test_metrics['roc_auc']}")
"""

NB07_M5 = """## 4. Results

Training curves, confusion matrices, and an ROC curve on the single held-out
test subject. With only a handful of windows per recording and a few
epochs, don't expect strong accuracy here — this notebook favors a fast,
honest, reproducible run over a flattering one; see the README's
reproducibility section and `reports/real_stew_status.md` for the caveats
that apply to any STEW accuracy number."""

NB07_C5 = """fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, result in results.items():
    epochs = [h["epoch"] for h in result["history"]]
    axes[0].plot(epochs, [h["train_loss"] for h in result["history"]], label=name)
    axes[1].plot(epochs, [h["val_accuracy"] for h in result["history"]], label=name)
axes[0].set(xlabel="epoch", ylabel="train loss", title="Training loss")
axes[1].set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy")
for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()

for name, result in results.items():
    cm = np.array(result["test_metrics"]["confusion_matrix"])
    print(f"{name} test confusion matrix (rows=true [lo,hi], cols=predicted):\\n{cm}")
"""

NB07_C6 = """fig, ax = plt.subplots(figsize=(5, 4))
target = ytest.numpy()
for name, result in results.items():
    probability = np.array(result["probability"])
    if len(set(target.tolist())) == 2:
        fpr, tpr, _ = roc_curve(target, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
ax.set(xlabel="false positive rate", ylabel="true positive rate", title="Test ROC (one held-out subject)")
ax.legend()
fig.tight_layout()
plt.show()
"""

NB07_M6 = """## Try it yourself

- Increase `--windows-per-record` / `epochs` in `run_experiment`, or switch
  to `configs/stew_full.yaml` + `hmb_kuramoto_ode.cli cross-validate` for the
  full grouped cross-validation protocol.
- Compare `HierarchicalKuramotoODE` ablation switches (`grand=False`,
  `kuramoto=False`, ...) on real data, the way notebook 06 does on synthetic
  data.
- Load `ratings.txt` (`REPO_DIR/ratings.txt`, columns `subject_id,
  rating_lo, rating_hi`) and check whether a subject's self-reported
  workload gap correlates with how confidently the model separates their
  `hi`/`lo` windows — remember it was never used for training or labels."""

NB07_CELLS = [M(NB07_INTRO), C(SETUP_CODE_FULL), M(NB07_M1), M(NB07_M2), C(NB07_C1), M(NB07_M3), C(NB07_C2), C(NB07_C3),
              M(NB07_M4), C(NB07_C4), M(NB07_M5), C(NB07_C5), C(NB07_C6), M(NB07_M6)]


# ======================================================================
# Write everything out
# ======================================================================
os.makedirs(OUT_DIR, exist_ok=True)
write_notebook("01_message_passing.ipynb", "01 - Message passing", NB01_CELLS)
write_notebook("02_gat_attention.ipynb", "02 - GAT attention", NB02_CELLS)
write_notebook("03_grand_diffusion.ipynb", "03 - GRAND diffusion", NB03_CELLS)
write_notebook("04_kuramoto_sync.ipynb", "04 - Kuramoto synchronization", NB04_CELLS)
write_notebook("05_neural_ode.ipynb", "05 - Neural ODE integration", NB05_CELLS)
write_notebook("06_synthetic_multitask.ipynb", "06 - Synthetic multitask training", NB06_CELLS)
write_notebook("07_stew_real_experiment.ipynb", "07 - Real STEW experiment", NB07_CELLS)
print("done")
