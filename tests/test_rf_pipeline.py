"""Checks for the wideband pipeline (src/pipeline/): cutting a box out,
the tracker, the look-alike rules, and the whole chain on a synthetic capture."""
import numpy as np
import pytest

from src.config import DATASET_CLASS_TO_IDX as IDX, DATASET_CLASSES, REPO_ROOT
from src.data import impairments
from src.detect import scenes
from src.generators import jam, spread
from src.pipeline import extract, resolve
from src.pipeline.capture import make_capture
from src.pipeline.run import EnergyDetector, NetDetector, box_to_physical, process_capture
from src.pipeline.tracker import Detection, Track, Tracker, freq_iou

FS = extract.FS
C = len(DATASET_CLASSES)


def probs_for(*classes, p=0.9, k=4):
    out = np.full((k, C), 0.01, dtype=np.float32)
    for c in classes:
        out[:, IDX[c]] = p
    return out


def det(t0, t1, f_lo, f_hi, probs=None, **feat):
    f = dict(wraps=0, sideband_db=8.0, line_db=40.0, snr_db=10.0, power_cv=0.0)
    f.update(feat)
    return Detection(t0, t1, f_lo, f_hi, 0.9, probs if probs is not None else probs_for("BPSK"), f)


# --- cutting a box out ------------------------------------------------------------

def _tone_capture(rng, f0=300e3, snr_db=10, n=8192):
    x = np.exp(2j * np.pi * f0 * np.arange(n) / FS) * np.sqrt(10 ** (snr_db / 10))
    return x + impairments.noise(n, 1.0, rng)


def test_noise_variance_reads_the_floor():
    rng = np.random.default_rng(0)
    assert extract.noise_variance(impairments.noise(scenes.SCENE_SAMPLES, 2.0, rng)) == pytest.approx(2.0, rel=0.1)


def test_cut_centres_the_box_and_refills_flat_noise():
    rng = np.random.default_rng(1)
    iq = _tone_capture(rng)
    c = extract.cut(iq, 0, len(iq), 280e3, 320e3, 1.0, rng)
    spec = np.abs(np.fft.fft(c.filled)) ** 2
    assert abs(np.fft.fftfreq(len(c.filled), 1 / FS)[spec.argmax()]) < 2e3      # the tone sits at 0 Hz
    far = np.abs(np.fft.fftfreq(len(c.filled), 1 / FS)) > 200e3
    assert np.mean(spec[far]) / len(c.filled) == pytest.approx(1.0, rel=0.15)    # flat unit noise elsewhere
    w = extract.windows(c)
    assert w.shape == (len(iq) // 512, 512)


def test_classifier_windows_keep_the_box_at_its_own_frequency():
    """Training windows sit anywhere in the band, never at exactly 0 Hz, so the
    classifier gets the tone back at ~300 kHz (the cut itself stays centred)."""
    rng = np.random.default_rng(1)
    iq = _tone_capture(rng)
    c = extract.cut(iq, 0, len(iq), 280e3, 320e3, 1.0, rng)
    f = np.fft.fftfreq(512, 1 / FS)
    for w in extract.windows(c)[:4]:
        assert abs(f[np.argmax(np.abs(np.fft.fft(w)))] - 300e3) < 12.5e3


def test_short_box_grows_to_one_window():
    rng = np.random.default_rng(2)
    iq = _tone_capture(rng)
    c = extract.cut(iq, 1000, 1100, 280e3, 320e3, 1.0, rng)
    assert c.t1 - c.t0 == 512 and len(extract.windows(c)) == 1


def test_features_tell_tone_from_fm_and_lora_from_sweep():
    rng = np.random.default_rng(3)
    n = 16384
    tone = extract.cut(_tone_capture(rng, 0.0, 10, n), 0, n, -40e3, 40e3, 1.0, rng).features
    fm = np.exp(1j * 2 * np.pi * np.cumsum(10e3 * np.sin(2 * np.pi * 20e3 * np.arange(n) / FS)) / FS)
    fm = extract.cut(fm * np.sqrt(10) + impairments.noise(n, 1.0, rng), 0, n, -60e3, 60e3, 1.0, rng).features
    assert tone["sideband_db"] < resolve.SIDEBAND_DB <= fm["sideband_db"]
    lora, _ = spread.generate_burst("LORA_CSS", n, rng)
    lo, hi = scenes.band_edges(lora)
    f = extract.cut(np.asarray(lora) * np.sqrt(10) + impairments.noise(n, 1.0, rng), 0, n, lo, hi, 1.0, rng).features
    assert f["wraps"] >= resolve.LORA_MIN_WRAPS


# --- tracker ---------------------------------------------------------------------------

def test_freq_iou():
    assert freq_iou(0, 10, 0, 10) == 1
    assert freq_iou(0, 10, 5, 15) == pytest.approx(5 / 15)
    assert freq_iou(0, 10, 20, 30) == 0


def test_tracker_joins_boxes_across_chunks_and_splits_by_frequency():
    tr = Tracker()
    tr.update([det(0, 32768, 100e3, 200e3), det(0, 32768, -500e3, -300e3)])
    tr.update([det(32768, 65536, 105e3, 205e3), det(32768, 40000, 900e3, 950e3)])
    tracks = tr.finish()
    assert len(tracks) == 3
    joined = [t for t in tracks if len(t.dets) == 2]
    assert len(joined) == 1 and joined[0].f_lo == pytest.approx(102.5e3)


def test_two_boxes_for_one_emitter_make_one_track_but_a_tone_on_top_stays_apart():
    tr = Tracker()
    tr.update([det(0, 32768, -500e3, 700e3), det(2000, 30000, -500e3, -40e3),      # wide signal + twin box
               det(5000, 30000, 100e3, 175e3)])                                    # narrow tone on top
    tracks = tr.finish()
    assert sorted(len(t.dets) for t in tracks) == [1, 2]


def test_tracker_starts_a_new_track_when_a_different_emitter_takes_the_band():
    def run(tr):
        for k, c in enumerate(("BPSK", "BPSK", "JAM_BARRAGE")):    # the third chunk is a jammer in the same band
            tr.update([det(k * 32768, (k + 1) * 32768, 100e3, 200e3, probs_for(c))])
        return sorted(len(t.dets) for t in tr.finish())
    assert run(Tracker()) == [1, 2]
    assert run(Tracker(min_class_overlap=None)) == [3]                 # frequency only: all joined


def test_tracker_closes_a_track_after_a_gap():
    tr = Tracker(max_gap_s=1e-3)
    tr.update([det(0, 1000, 0, 100e3)])
    tr.update([det(1000 + int(5e-3 * FS), 20000, 0, 100e3)])
    assert len(tr.finish()) == 2


def test_track_scores_average_in_log_odds():
    t = Track(1, [det(0, 512, 0, 1, probs_for("BPSK", p=0.9, k=3)), det(512, 1024, 0, 1, probs_for("BPSK", p=0.6, k=1))])
    p = t.probs()[IDX["BPSK"]]
    logit = (3 * np.log(9) + np.log(1.5)) / 4
    assert p == pytest.approx(1 / (1 + np.exp(-logit)), rel=1e-4)


# --- look-alike rules -----------------------------------------------------------------

def one(d):
    return Track(1, [d])


def test_wide_sweep_is_the_jammer_and_wrapping_narrow_one_is_lora():
    both = probs_for("LORA_CSS", "JAM_SWEPT")
    assert resolve.resolve(one(det(0, 32768, 0, 1.1e6, both))).label == "JAM_SWEPT"
    assert resolve.resolve(one(det(0, 32768, 0, 400e3, both, wraps=12))).label == "LORA_CSS"
    assert resolve.resolve(one(det(0, 32768, 0, 400e3, both, wraps=0))).label == "JAM_SWEPT"
    assert resolve.resolve(one(det(0, 1024, 0, 400e3, both, wraps=0))).label == "LORA_CSS / JAM_SWEPT"


def test_tone_with_sidebands_is_the_modulated_class():
    p = probs_for("RADAR_CW", "JAM_TONE", p=0.8)
    p[:, IDX["NBFM"]] = 0.5
    a = resolve.resolve(one(det(0, 32768, 0, 100e3, p, sideband_db=30)))
    assert a.label == "NBFM"


def test_pure_tone_cases():
    tones = probs_for("RADAR_CW", "JAM_TONE")
    lone = one(det(0, 32768, 0, 75e3, tones))
    assert resolve.resolve(lone).label == "RADAR_CW / JAM_TONE"
    assert resolve.resolve(lone, cw_min_s=5e-3).label == "RADAR_CW"
    victim = Track(1, [det(0, 65536, -100e3, 200e3, probs_for("BPSK"))])
    tone = Track(2, [det(32768, 65536, 0, 75e3, tones)])
    assert resolve.resolve(tone, [victim, tone]).label == "JAM_TONE"


def test_fm_classifier_answer_on_a_pure_strong_tone_becomes_a_tone():
    a = resolve.resolve(one(det(0, 32768, 0, 75e3, probs_for("NBFM"), sideband_db=9, line_db=50)))
    assert a.classes == ("RADAR_CW", "JAM_TONE")
    weak = resolve.resolve(one(det(0, 32768, 0, 75e3, probs_for("NBFM"), sideband_db=9, line_db=20)))
    assert weak.label == "NBFM"           # too weak to see sidebands: keep the classifier's answer


def test_size_groups_answer_the_group_not_a_size():
    p = probs_for("256QAM", p=0.6)
    p[:, IDX["64QAM"]] = 0.5
    a = resolve.resolve(one(det(0, 32768, 0, 200e3, p)))
    assert a.label == "QAM" and a.classes == resolve.SIZE_GROUPS["QAM"]
    assert "256QAM" in a.reason and a.confidence == pytest.approx(0.6, abs=1e-3)
    assert resolve.resolve(one(det(0, 32768, 0, 200e3, probs_for("16PSK")))).label == "8PSK / 16PSK / 32PSK"
    assert resolve.resolve(one(det(0, 32768, 0, 200e3, probs_for("8ASK")))).classes == ("4ASK", "8ASK")


def test_sizes_the_model_can_tell_apart_stay_exact():
    for c in ("BPSK", "QPSK", "OQPSK", "OOK", "4FSK"):
        assert resolve.resolve(one(det(0, 32768, 0, 200e3, probs_for(c)))).label == c


def test_size_groups_cover_every_member_once():
    from src.config import CLASS_GROUPS
    members = [c for m in resolve.SIZE_GROUPS.values() for c in m]
    assert len(members) == len(set(members))
    assert set(CLASS_GROUPS["QAM"]) | set(CLASS_GROUPS["APSK"]) <= set(members)
    assert set(members) <= set(DATASET_CLASSES)


def test_drfm_names_the_copied_radar():
    a = resolve.resolve(one(det(0, 32768, 0, 1e6, probs_for("JAM_DRFM", "RADAR_BARKER"))))
    assert a.label == "JAM_DRFM" and a.classes == ("JAM_DRFM", "RADAR_BARKER")


# --- whole chain --------------------------------------------------------------------

def test_box_to_physical():
    t0, t1, f_lo, f_hi = box_to_physical([10, 64, 20, 96], 32768)
    assert (t0, t1) == (32768 + 1280, 32768 + 2560)
    assert (f_lo, f_hi) == (0.0, 800e3)


def test_capture_truth_matches_its_signal():
    rng = np.random.default_rng(4)
    iq, truth = make_capture(rng, n_chunks=2, n_emitters=2, classes=["LORA_CSS"], snr_db=10)
    assert len(iq) == 2 * scenes.SCENE_SAMPLES and len(truth) == 2
    assert all(t["cls"] == "LORA_CSS" and t["t_end"] > t["t_start"] for t in truth)


def test_whole_chain_tracks_a_long_lora_burst():
    """A LoRa burst over 3 chunks: the energy detector boxes it per chunk, the
    tracker joins them into one track, and the rules settle the pair."""
    rng = np.random.default_rng(5)
    n = 3 * scenes.SCENE_SAMPLES
    sig, _ = spread.generate_burst("LORA_CSS", n - 4096, rng)
    iq = np.zeros(n, complex)
    iq[2048:n - 2048] = impairments.frequency_shift(np.asarray(sig), 400e3) * np.sqrt(10 ** 1.5)
    iq += impairments.noise(n, 1.0, rng)

    def stub(windows):          # a classifier that, like the real one, names both sweep classes
        return probs_for("LORA_CSS", "JAM_SWEPT", k=len(windows))

    answers, tracker = process_capture(iq, EnergyDetector(), stub, rng=rng)
    main = max(answers, key=lambda a: a["n_windows"])
    assert main["n_boxes"] >= 3 and main["t_end_ms"] - main["t_start_ms"] > 20
    assert main["answer"] == "LORA_CSS"
    assert 200 < (main["f_lo_khz"] + main["f_hi_khz"]) / 2 < 600


def test_swept_jammer_is_not_called_lora():
    rng = np.random.default_rng(6)
    n = 2 * scenes.SCENE_SAMPLES
    sig, _ = jam.generate_burst("JAM_SWEPT", n, rng)
    iq = np.asarray(sig) * np.sqrt(10) + impairments.noise(n, 1.0, rng)
    answers, _ = process_capture(iq, EnergyDetector(), lambda w: probs_for("LORA_CSS", "JAM_SWEPT", k=len(w)), rng=rng)
    assert max(answers, key=lambda a: a["n_windows"])["answer"] == "JAM_SWEPT"


DETECTOR = REPO_ROOT / "results" / "detector" / "emitter_detector.pt"


@pytest.mark.skipif(not DETECTOR.exists(), reason="no trained detector in results/detector/")
def test_trained_detector_runs_in_the_chain():
    rng = np.random.default_rng(7)
    iq, _ = make_capture(rng, n_chunks=2, n_emitters=2, snr_db=15)
    answers, _ = process_capture(iq, NetDetector(DETECTOR), lambda w: probs_for("BPSK", k=len(w)), rng=rng)
    assert all(a["answer"] == "BPSK" for a in answers)
