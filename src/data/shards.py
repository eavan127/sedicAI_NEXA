"""
Write, read and check data shards -- the one format every teammate's data is
handed over in (README "Shared data workflow").

A shard is an .npz with exactly these keys:

  iq      complex64 (N, window_len)   the windows
  y       uint8 (N, len(DATASET_CLASSES))  multi-hot, DATASET_CLASSES order
  snr_db  float32 (N,)
  source  str (N,)                    "gen" or the external dataset's name
  seed    int64 (N,)                  per-window seed for "gen" rows (any one
                                      window can be regenerated from it); the
                                      row index in the source file otherwise

Every shard is written together with a manifest JSON (same base name, in
data/manifests/) that records where it came from and its SHA-256, so the merge
can tell a complete, untouched shard from a half-synced or edited one.

ShardWriter is shared on purpose: if each teammate wrote their own .npz code,
the merge would be the first place a wrong dtype or column order showed up.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import CFG, DATASET_CLASSES, DATASET_CLASS_TO_IDX, REPO_ROOT

WINDOW_LEN = CFG["signal"]["window_len"]
MAX_PER_SHARD = 50_000
SPLITS = ("train", "val", "test")
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
SHARD_KEYS = ("iq", "y", "snr_db", "source", "seed")
# Fixed width so np.load never needs allow_pickle (object arrays would).
SOURCE_DTYPE = "<U24"


def class_list_version():
    """Short fingerprint of DATASET_CLASSES. Two shards with different values
    have different label columns and must never be merged."""
    return hashlib.sha256("\n".join(DATASET_CLASSES).encode()).hexdigest()[:12]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                               cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def multi_hot(class_names):
    vec = np.zeros(len(DATASET_CLASSES), dtype=np.uint8)
    for name in class_names:
        vec[DATASET_CLASS_TO_IDX[name]] = 1
    return vec


def check_contract(data):
    """Raise ValueError naming the first way `data` (a dict or NpzFile) breaks
    the shard contract."""
    missing = [k for k in SHARD_KEYS if k not in data]
    if missing:
        raise ValueError(f"shard is missing keys {missing}")
    iq, y = data["iq"], data["y"]
    n = len(iq)
    if iq.dtype != np.complex64 or iq.ndim != 2 or iq.shape[1] != WINDOW_LEN:
        raise ValueError(f"iq must be complex64 (N, {WINDOW_LEN}), got {iq.dtype} {iq.shape}")
    if y.dtype != np.uint8 or y.shape != (n, len(DATASET_CLASSES)):
        raise ValueError(f"y must be uint8 ({n}, {len(DATASET_CLASSES)}), got {y.dtype} {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1")
    for key, kind in (("snr_db", np.float32), ("seed", np.int64)):
        if data[key].dtype != kind or data[key].shape != (n,):
            raise ValueError(f"{key} must be {np.dtype(kind)} ({n},), got {data[key].dtype} {data[key].shape}")
    if data["source"].shape != (n,) or data["source"].dtype.kind != "U":
        raise ValueError(f"source must be a string array ({n},)")
    if not np.isfinite(iq).all():
        raise ValueError("iq contains NaN or inf")


class ShardWriter:
    """Collects windows and writes them as contract-checked shards of at most
    MAX_PER_SHARD windows, each with its manifest.

        with ShardWriter(out_dir, "fskpsk", "train", seed=4000) as w:
            w.add(iq_window, {"QPSK"}, snr_db=6, source="gen", seed=s)
    """

    def __init__(self, out_dir, group, split, seed, manifest_dir=MANIFEST_DIR,
                 max_per_shard=MAX_PER_SHARD, notes=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.out_dir, self.manifest_dir = Path(out_dir), Path(manifest_dir)
        self.group, self.split, self.seed = group, split, int(seed)
        self.max_per_shard, self.notes = max_per_shard, notes or {}
        self.part = 0
        self.written = []
        self._reset()

    def _reset(self):
        self._iq, self._y, self._snr, self._src, self._seed = [], [], [], [], []

    def add(self, iq, class_names, snr_db, source, seed):
        self._iq.append(np.asarray(iq, dtype=np.complex64))
        self._y.append(multi_hot(class_names))
        self._snr.append(snr_db)
        self._src.append(source)
        self._seed.append(seed)
        if len(self._iq) >= self.max_per_shard:
            self.flush()

    def flush(self):
        if not self._iq:
            return None
        data = {
            "iq": np.stack(self._iq).astype(np.complex64),
            "y": np.stack(self._y).astype(np.uint8),
            "snr_db": np.asarray(self._snr, dtype=np.float32),
            "source": np.asarray(self._src, dtype=SOURCE_DTYPE),
            "seed": np.asarray(self._seed, dtype=np.int64),
        }
        check_contract(data)
        name = f"{self.group}_{self.split}_seed{self.seed}_{self.part:03d}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.npz"
        np.savez(path, **data)
        commit, dirty = _git_commit()
        sources, src_counts = np.unique(data["source"], return_counts=True)
        manifest = {
            "shard": path.name,
            "group": self.group,
            "split": self.split,
            "seed": self.seed,
            "part": self.part,
            "n_windows": int(len(data["iq"])),
            "class_list_version": class_list_version(),
            "class_counts": {c: int(n) for c, n in zip(DATASET_CLASSES, data["y"].sum(axis=0)) if n},
            "source_counts": {s: int(n) for s, n in zip(sources, src_counts)},
            "snr_db": sorted({float(s) for s in data["snr_db"]}),
            "git_commit": commit,
            "git_dirty": dirty,
            "sha256": file_sha256(path),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": self.notes,
        }
        (self.manifest_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2))
        self.written.append(manifest)
        self.part += 1
        self._reset()
        return manifest

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.flush()


def load_shard(path, manifest=None):
    """Load a shard, checking the contract and (if a manifest is given) that
    the file is byte-for-byte the one the manifest describes."""
    path = Path(path)
    if manifest is not None:
        if manifest["class_list_version"] != class_list_version():
            raise ValueError(f"{path.name} was written for a different class list "
                             f"({manifest['class_list_version']} vs {class_list_version()})")
        actual = file_sha256(path)
        if actual != manifest["sha256"]:
            raise ValueError(f"{path.name} does not match its manifest SHA-256 "
                             "(half-synced or modified?)")
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in SHARD_KEYS}
    check_contract(data)
    return data
