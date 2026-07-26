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
