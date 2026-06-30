import random

import numpy as np
import torch
import transformers


def set_all_seeds(seed: int) -> None:
    """Set random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic behavior in PyTorch CUDA backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
