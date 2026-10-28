"""
Tests for the week-1 FSK / PSK / QAM generators, the shard format and the
external-dataset readers.

As with test_generators.py, these check INTERNAL CONSISTENCY -- that each
class is built the way src/generators/digital.py says it is -- not realism.
"""
import json

import numpy as np
import pytest

from src.config import DATASET_CLASSES
from src.data import external
from src.data.shards import ShardWriter, check_contract, load_shard
from src.generators import digital

N = digital.WINDOW_LEN


def inst_freq(sig):
    """Instantaneous frequency in cycles per sample."""
    return np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi)


# --- generators -------------------------------------------------------------

def test_every_generator_class_is_a_dataset_class():
    assert set(digital.CLASSES) <= set(DATASET_CLASSES)


@pytest.mark.parametrize("cls", digital.CLASSES)
def test_shape_dtype_and_unit_power(cls):
    sig, _ = digital.generate_window(cls, None, np.random.default_rng(0))
    assert sig.shape == (N,) and sig.dtype == np.complex64
    assert np.isfinite(sig).all()
    assert np.mean(np.abs(sig) ** 2) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("cls", digital.CLASSES)
def test_same_seed_same_window(cls):
    a, _ = digital.generate_window(cls, 4, np.random.default_rng(7))
    b, _ = digital.generate_window(cls, 4, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", ["QPSK", "64QAM", "GMSK", "8FSK"])
@pytest.mark.parametrize("snr_db", [-10, 0, 20])
def test_noise_is_added_at_the_requested_snr(cls, snr_db):
    """Same seed with and without noise gives the same signal, so the
    difference is exactly the noise that was added."""
    ratios = []
    for seed in range(20):
        clean, _ = digital.generate_window(cls, None, np.random.default_rng(seed))
        noisy, _ = digital.generate_window(cls, snr_db, np.random.default_rng(seed))
        noise = noisy.astype(complex) - clean
        ratios.append(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noise) ** 2))
    assert 10 * np.log10(np.mean(ratios)) == pytest.approx(snr_db, abs=0.5)


@pytest.mark.parametrize("cls,m", [("BPSK", 2), ("QPSK", 4), ("8PSK", 8), ("16PSK", 16),
                                   ("32PSK", 32), ("16QAM", 16), ("32QAM", 32),
                                   ("64QAM", 64), ("128QAM", 128), ("256QAM", 256)])
def test_constellation_has_the_named_number_of_points(cls, m):
    pts = digital.CONSTELLATIONS[cls]
    assert len(np.unique(np.round(pts, 6))) == m
    assert np.mean(np.abs(pts) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("cls", ["BPSK", "QPSK", "8PSK", "16PSK", "32PSK"])
def test_psk_points_all_lie_on_one_circle(cls):
    assert np.ptp(np.abs(digital.CONSTELLATIONS[cls])) < 1e-9


@pytest.mark.parametrize("cls", ["16QAM", "64QAM", "256QAM"])
def test_square_qam_is_a_grid(cls):
    pts = digital.CONSTELLATIONS[cls]
    side = int(np.sqrt(len(pts)))
    assert len(np.unique(np.round(pts.real, 6))) == side
    assert len(np.unique(np.round(pts.imag, 6))) == side


@pytest.mark.parametrize("cls", digital.FSK_CLASSES)
def test_fsk_family_has_constant_envelope(cls):
    sig, _ = digital.generate_window(cls, None, np.random.default_rng(1))
    assert np.ptp(np.abs(sig)) < 1e-3


@pytest.mark.parametrize("cls,m", [("2FSK", 2), ("4FSK", 4), ("8FSK", 8),
                                   ("CPFSK", 2), ("MSK", 2)])
def test_rectangular_fsk_uses_exactly_m_frequencies(cls, m):
    """Mid-symbol, a rectangular-pulse FSK sits on one of M tones."""
    sig, p = digital.generate_window(cls, None, np.random.default_rng(3))
    f = inst_freq(sig)
    sps, h = int(p["sps"]), p["h"]
    allowed = np.arange(-(m - 1), m, 2) * h / (2 * sps)
    # Ignore the sample that straddles each symbol change.
    nearest = np.min(np.abs(f[:, None] - allowed[None, :]), axis=1)
    assert np.mean(nearest < 1e-4) > 0.8
    used = {int(np.argmin(np.abs(allowed - x))) for x in f if np.min(np.abs(allowed - x)) < 1e-4}
    assert len(used) == m  # a 512-sample window visits every tone


@pytest.mark.parametrize("cls", ["MSK", "GMSK"])
def test_msk_family_deviation_is_quarter_symbol_rate(cls):
    sig, p = digital.generate_window(cls, None, np.random.default_rng(4))
    assert p["h"] == 0.5
    peak = np.max(np.abs(inst_freq(sig)))
    assert peak == pytest.approx(1 / (4 * p["sps"]), rel=0.02)


def test_gaussian_smoothing_removes_frequency_jumps():
    """GMSK's frequency moves smoothly; MSK's jumps between its two tones."""
    def biggest_step(cls):
        sig, _ = digital.generate_window(cls, None, np.random.default_rng(5))
        return np.max(np.abs(np.diff(inst_freq(sig))))
    assert biggest_step("GMSK") < 0.5 * biggest_step("MSK")


def test_continuous_phase_vs_phase_jumps():
    """CPFSK never jumps in phase; 2FSK restarts each tone at a random phase."""
    def biggest_phase_step(cls):
        sig, _ = digital.generate_window(cls, None, np.random.default_rng(6))
        return np.max(np.abs(np.angle(sig[1:] * np.conj(sig[:-1]))))
    assert biggest_phase_step("CPFSK") < 0.5
    assert biggest_phase_step("2FSK") > 1.0


@pytest.mark.parametrize("cls", digital.FSK_CLASSES)
def test_fsk_never_aliases(cls):
    for seed in range(30):
        sig, _ = digital.generate_window(cls, None, np.random.default_rng(seed))
        assert np.max(np.abs(inst_freq(sig))) < 0.5


def test_oqpsk_envelope_never_reaches_zero_but_qpsk_does():
    def min_env(cls):
        return min(np.min(np.abs(digital.generate_window(cls, None, np.random.default_rng(s))[0]))
                   for s in range(10))
    assert min_env("OQPSK") > 2 * min_env("QPSK")


# --- shards -----------------------------------------------------------------

def _write_small(tmp_path, n=5):
    rng = np.random.default_rng(0)
    with ShardWriter(tmp_path / "shards", "fskpsk", "train", seed=4000,
                     manifest_dir=tmp_path / "manifests", max_per_shard=3) as w:
        for i in range(n):
            iq, _ = digital.generate_window("QPSK", 6, rng)
            w.add(iq, {"QPSK"}, 6, "gen", i)
    return w


def test_shard_round_trip_and_manifest(tmp_path):
    w = _write_small(tmp_path)
    assert [m["n_windows"] for m in w.written] == [3, 2]  # split at max_per_shard
    m = json.loads((tmp_path / "manifests" / "fskpsk_train_seed4000_000.json").read_text())
    assert m["class_counts"] == {"QPSK": 3} and m["source_counts"] == {"gen": 3}
    data = load_shard(tmp_path / "shards" / m["shard"], manifest=m)
    assert data["y"][:, DATASET_CLASSES.index("QPSK")].all()
    assert data["y"].sum() == 3


def test_a_modified_shard_fails_its_checksum(tmp_path):
    w = _write_small(tmp_path)
    path = tmp_path / "shards" / w.written[0]["shard"]
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte inside the iq data
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        load_shard(path, manifest=w.written[0])


def test_contract_rejects_wrong_shapes():
    good = {"iq": np.zeros((2, N), np.complex64), "y": np.zeros((2, len(DATASET_CLASSES)), np.uint8),
            "snr_db": np.zeros(2, np.float32), "source": np.array(["gen", "gen"]),
            "seed": np.zeros(2, np.int64)}
    check_contract(good)
    for key, bad in (("iq", np.zeros((2, N), np.complex128)),
                     ("iq", np.zeros((2, N - 1), np.complex64)),
                     ("y", np.zeros((2, 8), np.uint8)),
                     ("snr_db", np.zeros(2, np.float64))):
        with pytest.raises(ValueError):
            check_contract({**good, key: bad})


# --- external readers (tiny fake files in the real layouts) -----------------

def test_radioml_reader_takes_disjoint_rows_and_first_512_samples(tmp_path):
    import h5py
    per_block, n_snr = 12, external.RADIOML_N_SNR_BINS
    n_rows = len(external.RADIOML_CLASS_ORDER) * n_snr * per_block
    X = np.zeros((n_rows, 1024, 2), np.float32)
    X[:, :, 0] = np.arange(n_rows)[:, None]          # I = global row number
    X[:, :, 1] = np.arange(1024)[None, :]            # Q = sample index
    with h5py.File(tmp_path / "fake.hdf5", "w") as f:
        f["X"] = X
    out = external.radioml_windows(["QPSK", "CPFSK"], [-10, 30, 40], {"train": 4, "val": 2, "test": 2},
                                   seed=1, path=tmp_path / "fake.hdf5", examples_per_block=per_block)
    assert not any(k[0] == "CPFSK" for k in out)      # not in RadioML
    assert not any(k[1] == 40 for k in out)           # outside RadioML's range
    rows = {split: out[("QPSK", -10, split)][1] for split in ("train", "val", "test")}
    assert len(set(np.concatenate(list(rows.values())))) == 8   # no row reused
    iq, r = out[("QPSK", -10, "train")]
    assert iq.shape == (4, N)
    np.testing.assert_array_equal(iq.real[:, 0], r)                 # the rows it says
    np.testing.assert_array_equal(iq.imag[0], np.arange(N))          # first 512 samples
    qpsk = external.RADIOML_CLASS_ORDER.index("QPSK")
    block = (qpsk * n_snr + (-10 + 20) // 2) * per_block
    assert ((r >= block) & (r < block + per_block)).all()           # right class + SNR


def test_hisarmod_reader_parses_matlab_complex_and_keeps_train_apart(tmp_path):
    root = tmp_path / "HisarMod2019.1"
    for folder, prefix, offset in (("Train", "train", 0), ("Test", "test", 1000)):
        d = root / folder
        d.mkdir(parents=True)
        labels = [2, 12, 2, 2, 22, 2, 2]
        snrs = [0, 0, 0, 4, 0, 0, 0]
        (d / f"{prefix}_labels.csv").write_text("\n".join(map(str, labels)) + "\n")
        (d / f"{prefix}_snr.csv").write_text("\n".join(map(str, snrs)) + "\n")
        rows = [",".join(f"{offset + r}.5-{k}i" for k in range(1024)) for r in range(len(labels))]
        (d / f"{prefix}_data.csv").write_text("\n".join(rows) + "\n")
    out = external.hisarmod_windows(["2FSK"], [0], {"train": 2, "val": 1, "test": 1}, seed=0, root=root)
    tr_iq, tr_rows = out[("2FSK", 0, "train")]
    assert set(tr_rows) <= {0, 2, 5, 6} and len(tr_rows) == 2       # label 2 at 0 dB only
    np.testing.assert_allclose(tr_iq.real[:, 0], tr_rows + 0.5)      # from the Train file
    np.testing.assert_allclose(tr_iq.imag[0, :3], [0, -1, -2])       # "a-bi" parsed
    assert tr_iq.shape == (2, N)
    va_iq, va_rows = out[("2FSK", 0, "val")]
    te_iq, te_rows = out[("2FSK", 0, "test")]
    assert va_iq.real[0, 0] >= 1000 and te_iq.real[0, 0] >= 1000    # from the Test file
    assert set(va_rows).isdisjoint(te_rows)
