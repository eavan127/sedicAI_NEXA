"""The 60-class model in the browser: PhysicsFeaturesONNX (the real-arithmetic
copy of the torch.fft physics branch), its ONNX export, and the shipped
web/models/best_model.onnx against the class list the page reads."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.amc_cnn import AMC_CNN, PhysicsFeatures
from src.models.onnx_export import AMC_CNN_ONNX, PhysicsFeaturesONNX, compute_stft_mag

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"stft_freq_summary": False, "stft_keep_rows": True, "cumulant_features": False,
         "expert_features": True, "full_spectrum": True, "physics_features": True}
# Rounding-noise features (see PhysicsFeaturesONNX): the envelope-spectrum
# median (6) and the instantaneous-frequency kurtosis (8).
NOISY = {6, 8}


def _windows(n=64, seed=0):
    """Random noise plus a few structured signals (tone, QPSK-like, chirp),
    normalised like the model's input."""
    rng = np.random.default_rng(seed)
    t = np.arange(512)
    z = list(rng.standard_normal((n, 512)) + 1j * rng.standard_normal((n, 512)))
    z.append(np.exp(2j * np.pi * 0.07 * t) + 0.05 * rng.standard_normal(512))
    sym = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, 64)))
    z.append(np.repeat(sym, 8) + 0.1 * rng.standard_normal(512))
    z.append(np.exp(1j * np.pi * 0.5 * t ** 2 / 512))
    x = np.stack([[w.real, w.imag] for w in z])
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / x.std(axis=(1, 2), keepdims=True)
    return torch.tensor(x, dtype=torch.float32)


def test_mirror_computes_the_same_physics_features():
    src = PhysicsFeatures(512)
    mirror = PhysicsFeaturesONNX(src, src.chan)
    x = _windows()
    with torch.no_grad():
        a, b = src(x).numpy(), mirror(x).numpy()
    assert a.shape == b.shape == (len(x), 56)
    steady = [i for i in range(56) if i not in NOISY]
    assert np.abs(a[:, steady] - b[:, steady]).max() < 1e-2
    # the noisy two stay close: feature 6 is a peak / median ratio in dB that
    # reads 60-75 dB on these windows, and moves by well under 1 dB
    assert np.abs(a[:, 6] - b[:, 6]).max() < 1.0


def test_wrapper_matches_the_model_with_physics():
    torch.manual_seed(0)
    model = AMC_CNN(num_classes=60, input_len=512, stft_n_fft=64, **FLAGS).eval()
    wrapper = AMC_CNN_ONNX(model).eval()
    x = _windows()
    mag = compute_stft_mag(x, 64, 4, model.stft_branch.window)
    with torch.no_grad():
        ref = model(x)
        out, attn = wrapper(x, mag)
    assert out.shape == (len(x), 60) and attn.shape == (len(x), 512)
    assert (out - ref).abs().max() < 0.1
    assert (out.argmax(1) == ref.argmax(1)).float().mean() > 0.95


def test_physics_model_exports_and_runs_in_onnxruntime(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_onnx import export_single_file
    torch.manual_seed(1)
    model = AMC_CNN(num_classes=60, input_len=512, stft_n_fft=64, **FLAGS).eval()
    ckpt = tmp_path / "physics.pt"
    torch.save({"state_dict": model.state_dict(),
                "meta": {"classes": [f"C{i}" for i in range(60)], "input_len": 512, "stft_n_fft": 64,
                         "model_flags": FLAGS}}, ckpt)
    out = tmp_path / "physics.onnx"
    assert export_single_file(ckpt, out)
    assert not out.with_name(out.name + ".data").exists()          # one file: onnxruntime-web needs that
    assert out.stat().st_size < 12e6                                 # DFT tables stored once, not per use


def _meta():
    src = (ROOT / "web" / "model_meta.js").read_text(encoding="utf-8")
    get = lambda name: json.loads(re.search(rf"export const {name} = (.+?);\n", src, re.S).group(1))  # noqa: E731
    return {k: get(k) for k in ("CLASSES", "WINDOW_LEN", "STFT_N_FFT", "STFT_HOP", "MODEL", "TIER_OF")}


def test_shipped_web_model_matches_the_page_class_list():
    """web/models/best_model.onnx, web/model_meta.js and the config's 60-class
    list are one model: the page reads scores by position in CLASSES."""
    onnx = pytest.importorskip("onnx")
    from src.config import DATASET_CLASSES
    meta = _meta()
    assert meta["CLASSES"] == list(DATASET_CLASSES)
    assert set(meta["TIER_OF"]) == set(DATASET_CLASSES)
    m = onnx.load(str(ROOT / "web" / "models" / "best_model.onnx"))
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    inputs = {i.name: dims(i) for i in m.graph.input}
    assert dims(m.graph.output[0])[1] == len(meta["CLASSES"])
    assert inputs["iq"][1:] == [2, meta["WINDOW_LEN"]]
    frames = 1 + (meta["WINDOW_LEN"] - meta["STFT_N_FFT"]) // meta["STFT_HOP"]
    assert inputs["stft_mag"][1:] == [1, meta["STFT_N_FFT"], frames]
    assert not list((ROOT / "web" / "models").glob("ensemble_*.onnx"))   # 8-class era files gone


def test_shipped_web_model_is_the_pipeline_checkpoint():
    """When the checkpoint is here (git-ignored), the committed ONNX file is
    its export: same scores, up to the physics branch's rounding noise."""
    ckpt = ROOT / "results" / "classifier" / "physics_best.pt"
    if not ckpt.exists():
        pytest.skip("results/classifier/physics_best.pt not here (git-ignored)")
    ort = pytest.importorskip("onnxruntime")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["meta"]["combined_version"] == _meta()["MODEL"]["data_version"]
    model = AMC_CNN(num_classes=60, input_len=512, stft_n_fft=64, **ck["meta"]["model_flags"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x = _windows(n=128, seed=5)
    mag = compute_stft_mag(x, 64, 4, model.stft_branch.window)
    sess = ort.InferenceSession(str(ROOT / "web" / "models" / "best_model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"iq": x.numpy(), "stft_mag": mag.numpy()})[0]
    with torch.no_grad():
        ref = model(x).numpy()
    assert (got.argmax(1) == ref.argmax(1)).mean() > 0.98
    assert np.quantile(np.abs(got - ref).max(1), 0.99) < 0.25
