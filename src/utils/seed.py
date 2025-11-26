import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU & CUDA) to improve reproducibility.
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # For extra determinism (may slow things down a bit)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        # In case torch isn't properly installed in some environments
        print(f"[seed] Warning: could not fully set torch seeds: {e}")
