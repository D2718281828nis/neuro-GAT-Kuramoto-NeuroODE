import random
import numpy as np
import torch
def seed_everything(seed=7): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
