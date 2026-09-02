"""Export best_model.pt and the 5-member ensemble to ONNX, and verify each
exported .onnx file's output matches the original PyTorch checkpoint's
output on random inputs.

Usage:
    python scripts/export_onnx.py

Writes to results/onnx/: best_model.onnx, ensemble_0.onnx .. ensemble_4.onnx.
Exits non-zero if any export fails to match its source checkpoint, so a
green run is a real guarantee, not just "no exception was thrown".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.models.onnx_export import (AMC_CNN_ONNX, compute_stft_mag,
                                     stft_mag_frame_count)

CKPT_DIR = REPO_ROOT / CFG["paths"]["checkpoints"]
OUT_DIR = CKPT_DIR / "onnx"
WINDOW_LEN = CFG["signal"]["window_len"]
N_FFT, HOP = 16, 4          # must match STFTBranch's own defaults
ATOL = 1e-4                  # logits are unbounded reals; this is a tight bar


def _load_model(ckpt_path: Path) -> AMC_CNN:
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def _sample_inputs(batch: int, seed: int):
    rng = np.random.default_rng(seed)
    iq = torch.tensor(rng.standard_normal((batch, 2, WINDOW_LEN)),
                       dtype=torch.float32)
    return iq


def export_and_verify(ckpt_path: Path, out_path: Path) -> bool:
    """Returns True iff the exported .onnx file's output matches the
    original checkpoint's output within ATOL, for two different batch
    sizes and two random seeds (catches shape bugs static-shape export can
    hide, and makes sure it isn't a one-seed fluke)."""
    print(f"\n== {ckpt_path.name} -> {out_path.name} ==")
    model = _load_model(ckpt_path)
    wrapper = AMC_CNN_ONNX(model).eval()
    window = model.stft_branch.window

    trace_iq = _sample_inputs(batch=2, seed=0)
    trace_mag = compute_stft_mag(trace_iq, N_FFT, HOP, window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (trace_iq, trace_mag), str(out_path),
        input_names=["iq", "stft_mag"], output_names=["logits", "attention"],
        dynamic_axes={"iq": {0: "batch"}, "stft_mag": {0: "batch"},
                       "logits": {0: "batch"}, "attention": {0: "batch"}},
        opset_version=18,  # torch 2.13's exporter targets 18 regardless;
                           # asking for 17 makes it noisily fail a downgrade
                           # attempt first, then fall back to 18 anyway.
    )
    print(f"  exported ({out_path.stat().st_size / 1024:.1f} KB)")

    sess = ort.InferenceSession(str(out_path),
                                 providers=["CPUExecutionProvider"])

    ok = True
    for batch, seed in [(1, 1), (8, 2)]:
        iq = _sample_inputs(batch, seed)
        mag = compute_stft_mag(iq, N_FFT, HOP, window)

        # Reference attention comes from the SAME forward hook
        # src/timeline.py:classify_capture uses, so this checks the inlined
        # copy in AMC_CNN_ONNX against the path the scorecard actually ran.
        captured = {}
        handle = model.attn_pool.score.register_forward_hook(
            lambda m, i, o: captured.__setitem__("scores", o.detach()))
        try:
            with torch.no_grad():
                expected = model(iq).numpy()
                expected_attn = torch.softmax(
                    captured["scores"], dim=2)[:, 0, :].numpy()
        finally:
            handle.remove()

        actual, actual_attn = sess.run(["logits", "attention"], {
            "iq": iq.numpy(), "stft_mag": mag.numpy(),
        })

        diff = np.abs(actual - expected).max()
        attn_diff = np.abs(actual_attn - expected_attn).max()
        passed = (np.allclose(actual, expected, atol=ATOL)
                   and np.allclose(actual_attn, expected_attn, atol=ATOL))
        ok = ok and passed
        status = "OK" if passed else "MISMATCH"
        print(f"  batch={batch} seed={seed}: max|diff| logits={diff:.2e} "
              f"attn={attn_diff:.2e}  [{status}]")

    return ok


def main():
    expected_frames = stft_mag_frame_count(WINDOW_LEN, N_FFT, HOP)
    print(f"window_len={WINDOW_LEN}  n_fft={N_FFT}  hop={HOP}  "
          f"-> {expected_frames} STFT frames")

    results = {}

    single = CKPT_DIR / "best_model.pt"
    results[single.name] = export_and_verify(single, OUT_DIR / "best_model.onnx")

    for i in range(5):
        ens = CKPT_DIR / f"ensemble_{i}.pt"
        if not ens.exists():
            print(f"\n== ensemble_{i}.pt: SKIPPED (not found) ==")
            continue
        results[ens.name] = export_and_verify(
            ens, OUT_DIR / f"ensemble_{i}.onnx")

    print("\n== summary ==")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nAt least one export did not match its source checkpoint.")
        sys.exit(1)
    print("\nAll exports match their source checkpoints.")


if __name__ == "__main__":
    main()
