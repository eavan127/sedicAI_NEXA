"""Loads the pipeline config so every module reads the same numbers.

Override the config file with the SEDIC_CONFIG env var, e.g. to run the whole
pipeline in seconds against configs/smoke.yaml before committing to a real run:

    SEDIC_CONFIG=configs/smoke.yaml python -m src.data.build_dataset
"""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path=None):
    path = path or os.environ.get("SEDIC_CONFIG") or DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()
CLASSES = CFG["classes"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def multi_hot(class_names):
    """Turn a set/iterable of class names into a (len(CLASSES),) 0/1 vector.

    Used for both standalone examples (one bit set) and composite/overlay
    examples (two or more bits set) -- the label representation is the same
    either way, just with more bits on for a composite example.
    """
    import numpy as np

    vec = np.zeros(len(CLASSES), dtype=np.float32)
    for name in class_names:
        vec[CLASS_TO_IDX[name]] = 1.0
    return vec


def resolve_multilabel_thresholds():
    """Per-class sigmoid decision threshold, as a (len(CLASSES),) array in
    CLASSES order. Falls back to `multilabel_threshold` for any class not
    listed in `multilabel_thresholds_per_class`.

    SHARED on purpose: every script that turns model probabilities into
    present/absent predictions (evaluate.py, train_ensemble.py,
    measure_variance.py) must call this instead of reading
    `multilabel_threshold` directly. They used to each have their own copy of
    that one-liner, which meant the per-class threshold fix that resolved the
    LFM_RADAR/FHSS benchmark FAIL only applied to evaluate.py -- the ensemble
    and variance scripts silently kept using the old single global 0.5
    fallback until this was centralised.
    """
    import numpy as np

    default = CFG.get("multilabel_threshold", 0.5)
    per_class = CFG.get("multilabel_thresholds_per_class", {})
    return np.array([per_class.get(c, default) for c in CLASSES], dtype=np.float32)
