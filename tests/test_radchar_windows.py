"""
Tests for src/data/radchar.py's radchar_windows -- the reader that hands the
five RadChar radar types to the week-1 shard script.

Runs against a tiny fake RadChar file built in tmp_path, laid out like the
real one (complex128 `iq`, structured `labels`), where every row's samples
equal its own row index -- so a returned window can be traced to the row it
claims to come from.
"""
import numpy as np
import pytest

from src.config import DATASET_CLASSES
from src.data.radchar import RADCHAR_CLASSES, RADCHAR_LEN, radchar_windows

h5py = pytest.importorskip("h5py")

TYPES = 5
SNRS = (-10, 0, 10)
PER_TYPE_SNR = 12
COUNTS = {"train": 6, "val": 2, "test": 2}


@pytest.fixture
def fake_radchar(tmp_path):
    rows = [(t, s) for t in range(TYPES) for s in SNRS for _ in range(PER_TYPE_SNR)]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)   # types and SNRs interleaved through the file, as in RadChar
    labels = np.zeros(len(rows), dtype=[("index", "<i8"), ("signal_type", "<i8"),
                                        ("signal_to_noise_ratio", "<i8")])
    labels["index"] = np.arange(len(rows))
    labels["signal_type"] = [t for t, _ in rows]
    labels["signal_to_noise_ratio"] = [s for _, s in rows]
    iq = np.repeat(np.arange(len(rows), dtype=np.complex128)[:, None], RADCHAR_LEN, axis=1)

    path = tmp_path / "RadChar-Fake.h5"
    with h5py.File(path, "w") as f:
        f["iq"] = iq
        f["labels"] = labels
    return path, labels


def test_class_names_are_dataset_classes():
    assert set(RADCHAR_CLASSES) <= set(DATASET_CLASSES)
    assert sorted(RADCHAR_CLASSES.values()) == list(range(TYPES))


def test_shape_dtype_and_rows_match_content(fake_radchar):
    path, labels = fake_radchar
    out = radchar_windows(list(RADCHAR_CLASSES), SNRS, COUNTS, seed=1, path=path)
    assert len(out) == TYPES * len(SNRS) * len(COUNTS)
    for (cls, snr, split), (iq, rows) in out.items():
        assert iq.dtype == np.complex64 and iq.shape == (COUNTS[split], RADCHAR_LEN)
        assert rows.dtype == np.int64 and (np.diff(rows) > 0).all()
        assert (iq.real[:, 0] == rows).all() and (iq.real == iq.real[:, :1]).all()
        assert (labels["signal_type"][rows] == RADCHAR_CLASSES[cls]).all()
        assert (labels["signal_to_noise_ratio"][rows] == snr).all()


def test_splits_never_share_a_row(fake_radchar):
    path, _ = fake_radchar
    out = radchar_windows(list(RADCHAR_CLASSES), SNRS, COUNTS, seed=1, path=path)
    for cls in RADCHAR_CLASSES:
        for snr in SNRS:
            parts = [set(out[(cls, snr, s)][1].tolist()) for s in COUNTS]
            assert sum(map(len, parts)) == len(set().union(*parts))


def test_missing_class_and_snr_are_absent(fake_radchar):
    path, _ = fake_radchar
    out = radchar_windows(["RADAR_COSTAS", "LFM_RADAR"], [10, 20], COUNTS, seed=1, path=path)
    assert {k[0] for k in out} == {"LFM_RADAR"}
    assert {k[1] for k in out} == {10}


def test_runs_out_gracefully_when_a_bin_is_small(fake_radchar):
    path, _ = fake_radchar
    out = radchar_windows(["RADAR_FRANK"], [0], {"train": 10, "val": 5, "test": 5}, seed=1, path=path)
    sizes = {s: len(out[("RADAR_FRANK", 0, s)][1]) for s in ("train", "val") if ("RADAR_FRANK", 0, s) in out}
    assert sizes == {"train": 10, "val": PER_TYPE_SNR - 10}
    assert ("RADAR_FRANK", 0, "test") not in out


def test_same_seed_same_rows_and_chunking_does_not_matter(fake_radchar):
    path, _ = fake_radchar
    a = radchar_windows(list(RADCHAR_CLASSES), SNRS, COUNTS, seed=7, path=path)
    b = radchar_windows(list(RADCHAR_CLASSES), SNRS, COUNTS, seed=7, path=path, chunk_rows=7)
    c = radchar_windows(list(RADCHAR_CLASSES), SNRS, COUNTS, seed=8, path=path)
    assert a.keys() == b.keys()
    for key in a:
        assert (a[key][1] == b[key][1]).all() and (a[key][0] == b[key][0]).all()
    assert any((a[k][1] != c[k][1]).any() for k in a)


def test_missing_file_says_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw/radchar"):
        radchar_windows(["LFM_RADAR"], [0], COUNTS, seed=1, path=tmp_path / "nope.h5")
