"""Checks for the noisy-voice simulator (src/intel/voice_sim.py): levels are what
they claim, the scoring is right, and the same seed gives the same audio."""
import shutil

import numpy as np
import pytest

from src.intel import voice_sim as vs


def speechy(seconds=3.0, seed=0):
    """A stand-in for speech: a band of noise switched on and off like syllables."""
    rng = np.random.default_rng(seed)
    n = int(seconds * vs.FS)
    x = vs.radio_band(rng.standard_normal(n))
    gate = np.sin(2 * np.pi * 3 * np.arange(n) / vs.FS) > 0
    return 0.2 * x * gate


def test_active_power_ignores_pauses():
    x = speechy()
    with_pause = np.concatenate([x, np.zeros(len(x))])
    assert vs.active_power(with_pause) == pytest.approx(vs.active_power(x), rel=0.05)


@pytest.mark.parametrize("snr", [20, 5, 0, -10])
def test_hiss_hits_the_requested_snr(snr):
    x = speechy()
    y = vs.with_hiss(x, snr, np.random.default_rng(1))
    noise = np.mean((y - x) ** 2)
    assert 10 * np.log10(vs.active_power(x) / noise) == pytest.approx(snr, abs=0.3)


@pytest.mark.parametrize("name", list(vs.SCENARIOS))
def test_scenarios_keep_length_and_never_clip(name):
    x, other = speechy(), speechy(1.0, seed=2)
    y = vs.simulate(name, x, other, np.random.default_rng(3))
    assert len(y) == len(x) and y.dtype == np.float32
    assert np.isfinite(y).all() and np.abs(y).max() <= 0.95 + 1e-6


@pytest.mark.parametrize("name", list(vs.SCENARIOS))
def test_same_seed_same_audio(name):
    x, other = speechy(), speechy(1.0, seed=2)
    a = vs.simulate(name, x, other, np.random.default_rng(4))
    b = vs.simulate(name, x, other, np.random.default_rng(4))
    assert np.array_equal(a, b)


def test_clean_is_untouched_and_noise_only_holds_no_speech():
    x, other = speechy(), speechy(1.0, seed=2)
    assert np.allclose(vs.simulate("clean", x, other, np.random.default_rng(0)), x, atol=1e-6)
    n = vs.simulate("noise_only", x, other, np.random.default_rng(99))    # speechy() itself used seed 0
    assert abs(np.corrcoef(n, x)[0, 1]) < 0.05


def test_second_voice_is_mixed_at_its_level():
    x, other = speechy(), speechy(1.0, seed=2)
    y = vs._talker(6)(x, other, np.random.default_rng(0))
    added = np.mean((y - x) ** 2)                 # the other talker plus 20 dB hiss
    assert added < vs.active_power(x)             # quieter than the main speech
    assert added > vs.active_power(x) / 10 ** 1.2  # but clearly present


def test_dropouts_keep_the_hiss_running_through_the_gaps():
    x = speechy(10.0)
    y = vs._dropouts(x, x, np.random.default_rng(0))
    assert not (np.abs(y) < 1e-12).any()


def test_normalize_handles_case_punctuation_and_digits():
    assert vs.normalize("Alpha 12, over.") == ["alpha", "one", "two", "over"]
    assert vs.normalize("Say again - it's OK") == ["say", "again", "its", "ok"]


def test_wer_counts_each_kind_of_error():
    assert vs.wer("alpha one over", "alpha one over")["wer"] == 0
    r = vs.wer("alpha one over", "alpha two over")
    assert (r["sub"], r["del"], r["ins"]) == (1, 0, 0) and r["wer"] == pytest.approx(1 / 3)
    r = vs.wer("alpha one over", "alpha over")
    assert (r["sub"], r["del"], r["ins"]) == (0, 1, 0)
    r = vs.wer("alpha over", "alpha one two over")
    assert (r["sub"], r["del"], r["ins"]) == (0, 0, 2) and r["wer"] == 1.0
    assert vs.wer("alpha 1", "Alpha one.")["wer"] == 0


def test_malay_digits_are_scored_as_malay_words():
    assert vs.normalize("Alfa 12, ganti.", "ms") == ["alfa", "satu", "dua", "ganti"]
    assert vs.wer("pusat pemeriksaan tujuh", "Pusat pemeriksaan 7.", "ms")["wer"] == 0
    assert vs.wer("pukul sembilan pagi", "pukul 9 pagi", "ms")["wer"] == 0
    assert vs.wer("pukul sembilan pagi", "pukul 9 pagi", "en")["wer"] > 0     # the language matters
    assert vs.wer("terputus-putus", "terputus putus", "ms")["wer"] == 0


def test_every_language_has_a_full_script():
    assert set(vs.SCRIPTS) == {"en", "ms"}
    assert len(vs.LINES_MS) == len(vs.LINES)
    assert all(vs.normalize(line, "ms") for line in vs.LINES_MS)


def test_recordings_pair_each_file_with_its_line(tmp_path):
    folder = tmp_path / "jessy_ms"
    folder.mkdir()
    x = 0.3 * np.sin(2 * np.pi * 300 * np.arange(vs.FS) / vs.FS)
    vs.save_wav(folder / "line2.wav", x)
    vs.save_wav(folder / "line5.wav", x)
    found = vs.recordings(folder, "ms")
    assert [label for _, _, label in found] == ["line2_jessy_ms", "line5_jessy_ms"]
    assert [line for _, line, _ in found] == [vs.LINES_MS[1], vs.LINES_MS[4]]
    assert vs.recordings(tmp_path / "missing", "ms") == []


def _fake_common_voice(root, speakers):
    """A tiny download in Common Voice's layout. speakers: {id: n clips}."""
    ms = root / "cv-corpus" / "ms"
    (ms / "clips").mkdir(parents=True)
    rows = ["client_id\tpath\tsentence_id\tsentence\tup_votes\tdown_votes"]
    x = 0.3 * np.sin(2 * np.pi * 300 * np.arange(vs.FS // 2) / vs.FS)
    for s, n in speakers.items():
        for i in range(n):
            vs.save_wav(ms / "clips" / f"{s}_{i}.wav", x)
            rows.append(f"{s}\t{s}_{i}.wav\t1\tAyat \"{s}\" nombor {i}.\t2\t0")
    rows.append("zz\tmissing.wav\t1\tFail tiada.\t2\t0")      # listed, but no audio on disk
    (ms / "validated.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_common_voice_spreads_picks_across_speakers(tmp_path):
    root = _fake_common_voice(tmp_path / "cv", {"a": 5, "b": 1, "c": 1})
    found = vs.common_voice(root, n=3, seed=0, cache_dir=tmp_path / "wav")
    assert [label for _, _, label in found] == ["cv01", "cv02", "cv03"]
    assert {line.split('"')[1] for _, line, _ in found} == {"a", "b", "c"}   # one each before anyone repeats
    assert all(len(x) == vs.FS // 2 for x, _, _ in found)
    assert all('"' in line for _, line, _ in found)                          # quotes in sentences survive
    every = vs.common_voice(root, n=99, seed=0, cache_dir=tmp_path / "wav")
    assert len(every) == 7                                                    # the listed-but-missing clip is skipped
    assert not any("zz" in label for _, _, label in every)


def test_common_voice_without_the_list_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="validated.tsv"):
        vs.common_voice(tmp_path, n=1)


def test_wer_on_no_speech():
    assert vs.wer("", "")["wer"] == 0
    r = vs.wer("", "thanks for watching")
    assert r["wer"] == 1 and r["hyp_words"] == 3


def test_wav_round_trip_and_resampling(tmp_path):
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(vs.FS) / vs.FS)
    vs.save_wav(tmp_path / "a.wav", x)
    y = vs.read_wav(tmp_path / "a.wav")
    assert len(y) == vs.FS and np.allclose(y, x, atol=1e-3)
    from src.intel.audio import write_wav
    write_wav(tmp_path / "b.wav", x[::2], 8000)
    assert abs(len(vs.read_wav(tmp_path / "b.wav")) - vs.FS) <= 2


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows voices only")
def test_windows_voice_speaks_offline(tmp_path):
    x = vs.spoken("Alpha one, this is base.", vs.VOICES[0], tmp_path)
    assert 0.5 < len(x) / vs.FS < 10 and vs.active_power(x) > 1e-5
