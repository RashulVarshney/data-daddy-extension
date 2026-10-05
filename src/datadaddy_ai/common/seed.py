import os
import random

import numpy as np

SEED = 1337


def set_global_seed(seed: int = SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def lib_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "faker",
        "rank-bm25",
        "sentence-transformers",
        "faiss-cpu",
        "torch",
        "huggingface_hub",
        "anthropic",
        "fastapi",
    ]
    out = {"python": os.sys.version.split()[0]}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not-installed"
    return out
