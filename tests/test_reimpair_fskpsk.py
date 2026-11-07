"""scripts/reimpair_fskpsk_shards.py: generated rows re-made with impairments
from their own seeds, external rows untouched, everything else kept."""
import json

import numpy as np
import pytest

from scripts.reimpair_fskpsk_shards import reimpair_split
from src.data import impairments
from src.data.shards import ShardWriter, load_shard
from src.generators import digital

FS = 3.2e6


def _old_style_shard(tmp_path, n=12):
    """What make_fskpsk_shards.py wrote before the fix: generate_window, no
    impairments, plus one external row."""
    rng = np.random.default_rng(0)
    ext = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    with ShardWriter(tmp_path / "old" / "fskpsk", "fskpsk", "test", 4002,
                     manifest_dir=tmp_path / "old_manifests") as w:
        for i in range(n):
            cls = ("BPSK", "GFSK", "16QAM")[i % 3]
            w.add(digital.generate_window(cls, 10.0, np.random.default_rng(1000 + i))[0], {cls}, 10.0, "gen", 1000 + i)
        w.add(ext, {"QPSK"}, 6.0, "radioml2018", 77)
    return w.written, ext


def _centre_hz(x):
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    return float((P * np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS))).sum() / P.sum())


def test_generated_rows_are_reimpaired_and_external_rows_kept(tmp_path):
    old_manifests, ext = _old_style_shard(tmp_path)
    stats, written = reimpair_split(old_manifests, tmp_path / "old" / "fskpsk", tmp_path / "new",
                                    tmp_path / "new_manifests", "test")
    assert stats["gen"] == 12 and stats["ext"] == 1 and stats["checked"] >= 1
    old = load_shard(tmp_path / "old" / "fskpsk" / old_manifests[0]["shard"])
    new = load_shard(tmp_path / "new" / written[0]["shard"],
                     json.loads((tmp_path / "new_manifests" / written[0]["shard"].replace(".npz", ".json")).read_text()))
    for k in ("y", "snr_db", "source", "seed"):
        assert np.array_equal(old[k], new[k]), k
    assert np.array_equal(new["iq"][-1], ext)                       # external row untouched
    for i in range(12):                                             # = what the fixed make script writes
        cls = ("BPSK", "GFSK", "16QAM")[i % 3]
        want, _ = impairments.generate_impaired(digital, cls, 10.0, np.random.default_rng(1000 + i))
        assert np.allclose(new["iq"][i], want, atol=1e-5)
    offsets = [abs(_centre_hz(x)) for x in new["iq"][:12]]
    assert np.median(offsets) > 50e3                                # no longer all at 0 Hz
    assert written[0]["notes"]["impairments"] is True


def test_a_seed_that_no_longer_reproduces_its_window_stops_the_run(tmp_path):
    rng = np.random.default_rng(0)
    with ShardWriter(tmp_path / "old" / "fskpsk", "fskpsk", "test", 4002,
                     manifest_dir=tmp_path / "old_manifests") as w:
        w.add(digital.generate_window("BPSK", 10.0, rng)[0], {"BPSK"}, 10.0, "gen", 5)   # stored seed is wrong
    with pytest.raises(ValueError, match="no longer reproduces"):
        reimpair_split(w.written, tmp_path / "old" / "fskpsk", tmp_path / "new", tmp_path / "nm", "test")
