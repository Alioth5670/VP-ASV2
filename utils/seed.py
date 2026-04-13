import os
import random
import numpy as np
import torch


def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    warn_only: bool = True,
) -> None:
    """Set random seed for python, numpy and torch.

    Args:
        seed: Global random seed.
        deterministic: Whether to enable deterministic algorithms for torch.
        benchmark: Value for torch.backends.cudnn.benchmark.
        warn_only: Passed to torch.use_deterministic_algorithms when deterministic=True.
    """
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        # Make matmul deterministic on CUDA (CUDA >= 10.2)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.use_deterministic_algorithms(False)
