"""Common lightweight baseline factory used by smoke comparisons."""
from torch import nn
BASELINES=("mlp","gcn","gat","grand","kuramoto","kuramoto_fixed","kuramoto_attention","kuramoto_grand_attention","full")
def baseline_names(): return BASELINES
def mlp_baseline(features=6,classes=2): return nn.Sequential(nn.Linear(features,16),nn.ReLU(),nn.Linear(16,classes))
