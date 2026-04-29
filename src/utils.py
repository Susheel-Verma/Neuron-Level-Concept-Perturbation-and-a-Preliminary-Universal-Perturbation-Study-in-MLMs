# Minimal utils stub for MM-Neurons
# This satisfies imports required for LLaVA experiments

import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
