"""Export best_model.pt and the 5-member ensemble to ONNX, and verify each
exported .onnx file's output matches the original PyTorch checkpoint's
output on random inputs.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --ckpt <week-1 best.pt> --out <file.onnx>

Writes to results/onnx/: best_model.onnx, ensemble_0.onnx .. ensemble_4.onnx.
With --ckpt it exports that one checkpoint instead: either a plain state_dict
(the 8-class models above) or a week-1 checkpoint ({"state_dict", "meta"},
written by scripts/train_week1.py), whose meta gives the class count, window
length, STFT n_fft and model flags (expert features, full spectrum, ...).
The --ckpt file is saved as ONE self-contained .onnx (no .onnx.data sidecar),
since onnxruntime-web loads a single file.
Exits non-zero if any export fails to match its source checkpoint, so a
green run is a real guarantee, not just "no exception was thrown".
"""
import argparse
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
ATOL = 1e-4                  # logits are unbounded reals; this is a tight bar
# A model with the physics branch is exported through PhysicsFeaturesONNX, a
# real-arithmetic copy of a branch that uses torch.fft (ONNX has no complex
# tensors). Two of its 56 features are rounding noise for some windows -- the
# median of the envelope spectrum of a constant-envelope signal is ~1e-5 of
# its peak, and the kurtosis of a near-constant frequency divides by a tiny
# spread -- so no float32 implementation, the ONNX one included, reproduces
# torch.fft's value there exactly. On real windows that moves a logit by a
# few hundredths. Such a model is therefore checked on what an operator sees:
# logits within PHYSICS_ATOL on random batches, attention (no physics in it)
# within ATOL, and on real windows the same top-1 class for at least
# PHYSICS_TOP1_AGREE of them, with the 99th-percentile logit change at most
# PHYSICS_MAX_LOGIT_DIFF.
PHYSICS_ATOL = 0.1
PHYSICS_TOP1_AGREE = 0.995
PHYSICS_MAX_LOGIT_DIFF = 0.25
REAL_WINDOWS = REPO_ROOT / "data" / "combined" / "test" / "X.npy"


def _load_model(ckpt_path: Path):
    """(model in eval mode, window_len). A week-1 checkpoint carries its own
    shape and flags in "meta"; a plain state_dict is one of the 8-class
    models, built from the config like before."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "meta" in ck and "state_dict" in ck:
        meta = ck["meta"]
        window_len = meta["input_len"]
        model = AMC_CNN(num_classes=len(meta["classes"]), input_len=window_len,
                        stft_n_fft=meta["stft_n_fft"], **meta.get("model_flags", {}))
        model.load_state_dict(ck["state_dict"])
    else:
        window_len = WINDOW_LEN
        model = AMC_CNN(num_classes=len(CLASSES), input_len=window_len)
        model.load_state_dict(ck)
    model.eval()
    return model, window_len


def _sample_inputs(batch: int, seed: int, window_len: int = WINDOW_LEN):
    rng = np.random.default_rng(seed)
    iq = torch.tensor(rng.standard_normal((batch, 2, window_len)),
                       dtype=torch.float32)
    return iq


def export_and_verify(ckpt_path: Path, out_path: Path) -> bool:
    """Returns True iff the exported .onnx file's output matches the
    original checkpoint's output within ATOL, for two different batch
    sizes and two random seeds (catches shape bugs static-shape export can
    hide, and makes sure it isn't a one-seed fluke)."""
    print(f"\n== {ckpt_path.name} -> {out_path.name} ==")
    model, window_len = _load_model(ckpt_path)
    wrapper = AMC_CNN_ONNX(model).eval()
    window = model.stft_branch.window
    n_fft, hop = model.stft_branch.n_fft, model.stft_branch.hop_length
    print(f"  window_len={window_len}  n_fft={n_fft}  hop={hop}  "
          f"-> {stft_mag_frame_count(window_len, n_fft, hop)} STFT frames")

    has_physics = getattr(model, "physics_branch", None) is not None
    trace_iq = _sample_inputs(batch=2, seed=0, window_len=window_len)
    trace_mag = compute_stft_mag(trace_iq, n_fft, hop, window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (trace_iq, trace_mag), str(out_path),
        input_names=["iq", "stft_mag"], output_names=["logits", "attention"],
        dynamic_axes={"iq": {0: "batch"}, "stft_mag": {0: "batch"},
                       "logits": {0: "batch"}, "attention": {0: "batch"}},
        opset_version=18,  # torch 2.13's exporter targets 18 regardless;
                           # asking for 17 makes it noisily fail a downgrade
                           # attempt first, then fall back to 18 anyway.
        # The physics branch's 305-lag loop and DFT matmuls make a graph the
        # default (dynamo) exporter took over 15 minutes on without finishing;
        # the TorchScript exporter does it in about half a minute.
        **({"dynamo": False} if has_physics else {}),
    )
    if has_physics:
        _dedupe_constants(out_path)
    print(f"  exported ({out_path.stat().st_size / 1024:.1f} KB)")

    sess = ort.InferenceSession(str(out_path),
                                 providers=["CPUExecutionProvider"])

    ok = True
    for batch, seed in [(1, 1), (8, 2)]:
        iq = _sample_inputs(batch, seed, window_len)
        mag = compute_stft_mag(iq, n_fft, hop, window)

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
        passed = (np.allclose(actual, expected, atol=PHYSICS_ATOL if has_physics else ATOL)
                   and np.allclose(actual_attn, expected_attn, atol=ATOL))
        ok = ok and passed
        status = "OK" if passed else "MISMATCH"
        print(f"  batch={batch} seed={seed}: max|diff| logits={diff:.2e} "
              f"attn={attn_diff:.2e}  [{status}]")

    if has_physics:
        ok = _check_physics_against_original(model, sess, window_len, n_fft, hop, window) and ok
    return ok


def _dedupe_constants(path: Path) -> None:
    """The TorchScript exporter writes a buffer as a Constant node at EVERY
    use, so the 512 x 512 DFT tables appear once per matrix product (54 MB
    for the physics model). Keep one copy of each identical tensor, as an
    initializer, and point every user at it."""
    import hashlib

    import onnx
    model = onnx.load(str(path))
    g = model.graph
    keep, rename, drop = {}, {}, []
    for node in g.node:
        if node.op_type != "Constant" or len(node.attribute) != 1 or node.attribute[0].name != "value":
            continue
        t = node.attribute[0].t
        if len(t.raw_data) < 4096:
            continue
        key = hashlib.sha256(t.raw_data + bytes(str((t.data_type, list(t.dims))), "ascii")).hexdigest()
        if key not in keep:
            init = onnx.TensorProto()
            init.CopyFrom(t)
            init.name = f"const_{len(keep)}"
            g.initializer.append(init)
            keep[key] = init.name
        rename[node.output[0]] = keep[key]
        drop.append(node)
    for node in drop:
        g.node.remove(node)
    for node in g.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
    onnx.save_model(model, str(path), save_as_external_data=False)
    print(f"  {len(drop)} large constants -> {len(keep)} shared initializers")


def _check_physics_against_original(model, sess, window_len, n_fft, hop, window, n=2048):
    """The ONNX file against the ORIGINAL model (torch.fft physics branch) on
    real windows when data/combined is here, else on random ones."""
    if REAL_WINDOWS.is_file():
        X = np.load(REAL_WINDOWS, mmap_mode="r")
        idx = np.sort(np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False))
        iq = torch.tensor(np.asarray(X[idx]), dtype=torch.float32)
        where = f"{len(idx)} real test windows (data/combined/test)"
    else:
        iq = _sample_inputs(n, 3, window_len)
        where = f"{n} random windows (data/combined not on this machine)"
    mag = compute_stft_mag(iq, n_fft, hop, window)
    with torch.no_grad():
        expected = torch.cat([model(iq[i:i + 256]) for i in range(0, len(iq), 256)]).numpy()
    actual = np.concatenate([sess.run(["logits"], {"iq": iq[i:i + 256].numpy(),
                                                     "stft_mag": mag[i:i + 256].numpy()})[0]
                              for i in range(0, len(iq), 256)])
    d = np.abs(actual - expected).max(axis=1)
    agree = float((actual.argmax(1) == expected.argmax(1)).mean())
    passed = agree >= PHYSICS_TOP1_AGREE and float(np.quantile(d, 0.99)) <= PHYSICS_MAX_LOGIT_DIFF
    print(f"  vs original model (torch.fft physics), {where}: top-1 agrees on {agree:.2%}, "
          f"logit |diff| median {np.median(d):.1e} p99 {np.quantile(d, 0.99):.1e} max {d.max():.1e}  "
          f"[{'OK' if passed else 'MISMATCH'}]")
    return passed


def export_single_file(ckpt_path: Path, out_path: Path) -> bool:
    """export_and_verify, then fold the weights into the .onnx itself."""
    import onnx
    if not export_and_verify(ckpt_path, out_path):
        return False
    onnx.save_model(onnx.load(str(out_path)), str(out_path), save_as_external_data=False)
    out_path.with_name(out_path.name + ".data").unlink(missing_ok=True)
    print(f"  saved as one file ({out_path.stat().st_size / 1024:.1f} KB)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, help="export just this checkpoint")
    p.add_argument("--out", type=Path, help="with --ckpt: output .onnx "
                   "(default: next to the checkpoint, same name)")
    args = p.parse_args()

    if args.ckpt is not None:
        out = args.out or args.ckpt.with_suffix(".onnx")
        if not export_single_file(args.ckpt, out):
            print()
            print("The export did not match its source checkpoint.")
            sys.exit(1)
        print()
        print(f"Export matches its source checkpoint: {out}")
        return

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
