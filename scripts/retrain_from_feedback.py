"""Retrain from human feedback: approved corrections -> candidate model.

Started by the local server when an expert clicks "Start retraining" on the
Model page (POST /api/retrain/start), or by hand:

    python scripts/retrain_from_feedback.py --job <job id>

The model is the 60-class one the dashboard and the Python Signal Pipeline
both run: configs/default.yaml browser_model (or the active retrained version
of it). What it does, in order (each step is printed, and the page shows the
log):

  1. Collect every APPROVED correction and cut its raw IQ into 512-sample
     windows labelled with what the human said was there. Pending and
     rejected corrections are never used, and neither is a label that names
     only the family "JAMMING" (no jammer type: what corrections said before
     the switch to 60 classes) -- the model has no class to learn it as.
  2. Mix them with a REPLAY sample of the original training data
     (data/combined/train, the 60-class training set): fine-tuning on
     corrections alone makes a model forget what it knew ("catastrophic
     forgetting").
  3. Fine-tune for a couple of epochs, corrections up-weighted, with the loss
     the model was trained with (scripts/train_week1.py).
  4. THE GATE: score the old and the new model on the same fixed exam -- a
     fixed sample of the held-out TEST split (data/combined/test), never
     trained on and never containing a correction. Classes are decided as the
     dashboard decides them (configs/default.yaml web_threshold). The new
     model passes only if no judged class (LFM_RADAR, FHSS, and JAMMING = any
     jammer type found on a jammer window) loses more than 1 point of recall,
     none falls below 80% unless the old one was already below it, and
     pure-noise false alarms do not rise by more than 0.5 points. Without
     data/combined (a teammate's machine), the exam is a fixed synthetic set
     of judged-class windows from the week-1 generators, and the report says
     so.
  5. Export the new model to ONNX (checked against PyTorch, see
     scripts/export_onnx.py) and register it as a CANDIDATE. Nothing changes
     for operators until an analyst approves it on the Model page.

Scope: "single" only. "both" (single + the 5-member ensemble) belonged to the
8-class model; there is no 60-class ensemble, so it is refused.

Nothing here overwrites a shipped file: every version is written to
data/local/models/<version>/ and kept forever, whether it passed or not. The
shipped checkpoint and web/models/best_model.onnx stay untouched, so
"use shipped" always restores them exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.localdb import CLASSES, FAMILY_LABELS, JAMMERS, JUDGED, LocalDB  # noqa: E402
from src.report_export import _iq_payload  # noqa: E402

FS = 3_200_000
WIN = 512
from src.config import CFG as _CFG  # noqa: E402
SHIPPED = ROOT / _CFG.get("browser_model", "results/classifier/physics_aug_best.pt")
COMBINED = ROOT / "data" / "combined"
JAM_IDX = [CLASSES.index(c) for c in JAMMERS]
NOISE_IDX = CLASSES.index("NOISE_FLOOR")
DTYPES = {"cf32_le": "<f4", "cf32_be": ">f4", "cf64_le": "<f8", "cf64_be": ">f8",
          "ci32_le": "<i4", "ci32_be": ">i4", "ci16_le": "<i2", "ci16_be": ">i2",
          "ci8": "i1", "cu8": "u1"}


def say(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def decode(datatype: str, data: bytes) -> np.ndarray:
    if datatype not in DTYPES:
        raise ValueError(f"unsupported datatype {datatype}")
    raw = np.frombuffer(data[: len(data) - len(data) % (2 * np.dtype(DTYPES[datatype]).itemsize)],
                        dtype=DTYPES[datatype]).astype(np.float64)
    if datatype == "cu8":
        raw = raw - 127.5
    return raw[0::2] + 1j * raw[1::2]


def multi_hot(labels) -> np.ndarray:
    v = np.zeros(len(CLASSES), np.float32)
    for c in labels:
        v[CLASSES.index(c)] = 1.0
    return v


def correction_windows(db: LocalDB, max_per_correction: int = 200):
    """Approved corrections -> (X, y, used ids, skipped ids, family-only ids).
    Windows are cut every 256 samples inside the corrected span; a span
    shorter than one window gets one window centred on it."""
    from src.data.preprocess import preprocess_window

    X, y, used, skipped, family_only = [], [], [], [], []
    cache = {}
    for c in db.list_corrections(status="approved", limit=100000):
        if any(lbl in FAMILY_LABELS for lbl in c["corrected_labels"]):
            family_only.append(c["id"])        # "JAMMING" without a type: nothing to learn it as
            continue
        key = c["iq_path"]
        if key not in cache:
            payload = _iq_payload(db.path.parent, key) if key else None
            if payload and abs(float(payload[1]) - FS) > 1:
                payload = None                      # other sample rates: the model never saw them
            cache[key] = decode(payload[0], payload[2]) if payload else None
        iq = cache[key]
        if iq is None or len(iq) < WIN:
            skipped.append(c["id"])
            continue
        a, b = int(c["start_s"] * FS), int(c["end_s"] * FS)
        if b - a < WIN:
            mid = (a + b) // 2
            starts = [min(max(mid - WIN // 2, 0), len(iq) - WIN)]
        else:
            starts = list(range(a, min(b, len(iq)) - WIN + 1, WIN // 2))[:max_per_correction]
        label = multi_hot(c["corrected_labels"])
        for s in starts:
            X.append(preprocess_window(iq[s:s + WIN], WIN))
            y.append(label)
        used.append(c["id"])
    if not X:
        return (np.zeros((0, 2, WIN), np.float32), np.zeros((0, len(CLASSES)), np.float32),
                used, skipped, family_only)
    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32), used, skipped, family_only


def dataset_available() -> bool:
    return all((COMBINED / split / f).exists() for split in ("train", "test") for f in ("X.npy", "y.npy"))


def dataset_split():
    """data/combined: (X_train, y_train, X_test, y_test), memory-mapped where
    possible (the training X is ~2 GB of float16)."""
    Xtr = np.load(COMBINED / "train" / "X.npy", mmap_mode="r")
    ytr = np.load(COMBINED / "train" / "y.npy", mmap_mode="r")
    Xte = np.load(COMBINED / "test" / "X.npy", mmap_mode="r")
    yte = np.load(COMBINED / "test" / "y.npy", mmap_mode="r")
    return Xtr, ytr, Xte, yte


def take(X, y, idx):
    idx = np.sort(np.asarray(idx))
    return np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.float32)


def synthetic_set(n_per_class: int, seed: int):
    """A FIXED synthetic exam of judged-class + noise windows (same seed ->
    same windows), from the week-1 generators the 60-class training data was
    made with, for machines without data/combined. Jammer types are spread
    over the jammer share, so every judged family gets n_per_class windows."""
    from src.data.preprocess import preprocess_window
    from src.generators import jam, radar, spread

    rng = np.random.default_rng(seed)
    plan = [("LFM_RADAR", radar, n_per_class), ("FHSS", spread, n_per_class)]
    plan += [(c, jam, max(n_per_class // len(JAMMERS), 1)) for c in JAMMERS]
    plan += [("NOISE_FLOOR", jam, n_per_class)]
    X, y = [], []
    for cls, gen, n in plan:
        for i in range(n):
            iq, _ = gen.generate_window(cls, [-6, -2, 2, 6, 10][i % 5], rng)
            X.append(preprocess_window(np.asarray(iq, dtype=complex), WIN))
            y.append(multi_hot([cls]))
    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32)


def predict(model, X, batch: int = 256) -> np.ndarray:
    """Sigmoid probabilities (the dashboard's and the pipeline's scores)."""
    import torch
    with torch.no_grad():
        out = [torch.sigmoid(model(torch.tensor(np.asarray(X[i:i + batch], dtype=np.float32)))).numpy()
               for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def score(model, X, y, thr) -> dict:
    """Recall of the judged classes as an operator sees them, and false
    alarms on pure noise. JAMMING is the family: a jammer window counts as
    found when ANY jammer type is reported on it (the type is a separate
    question the gate does not judge)."""
    p = predict(model, X) > thr
    yb = y > 0.5
    rec = {}
    for c in JUDGED:
        if c == "JAMMING":
            pos = yb[:, JAM_IDX].any(axis=1)
            hit = p[:, JAM_IDX].any(axis=1)
        else:
            j = CLASSES.index(c)
            pos, hit = yb[:, j], p[:, j]
        rec[c] = float(hit[pos].mean()) if pos.any() else None
    noise_only = yb[:, NOISE_IDX] & (yb.sum(axis=1) == 1)
    non_noise = [j for j in range(len(CLASSES)) if j != NOISE_IDX]
    fa = float(p[noise_only][:, non_noise].any(axis=1).mean()) if noise_only.any() else None
    return {"recall": rec, "noise_false_alarm": fa}


def exact_fit(model, X, y, thr) -> float | None:
    """Share of correction windows where the model now says exactly what the human said."""
    if not len(X):
        return None
    p = predict(model, X) > thr
    return float((p == (y > 0.5)).all(axis=1).mean())


def gate(before: dict, after: dict, max_drop: float = 0.01) -> tuple[bool, list]:
    rules = []
    for c in JUDGED:
        b, a = before["recall"][c], after["recall"][c]
        if b is None or a is None:
            continue
        rules.append({"text": f"{c}: recall may not drop more than {max_drop * 100:g} point(s) ({b:.3f} -> {a:.3f})",
                      "met": a >= b - max_drop - 1e-9})
        floor = min(0.80, b)
        rules.append({"text": f"{c}: recall stays at or above {floor:.2f} ({a:.3f})", "met": a >= floor - 1e-9})
    b, a = before["noise_false_alarm"], after["noise_false_alarm"]
    if b is not None and a is not None:
        rules.append({"text": f"False alarms on pure noise may not rise more than 0.5 points ({b:.4f} -> {a:.4f})",
                      "met": a <= b + 0.005})
    return all(r["met"] for r in rules) and bool(rules), rules


def export_onnx_file(pt: Path, onnx_path: Path) -> None:
    """ONNX, verified against PyTorch, as ONE self-contained file: torch writes
    the weights to a sidecar .onnx.data, which the browser's onnxruntime-web
    cannot load ("Module.MountedFiles is not available"). Fold them back in,
    as web/build.py does for the shipped models."""
    import onnx
    from export_onnx import export_and_verify
    if not export_and_verify(pt, onnx_path):
        raise RuntimeError(f"ONNX export of {pt.name} did not match PyTorch; candidate not registered")
    onnx.save_model(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)
    onnx_path.with_name(onnx_path.name + ".data").unlink(missing_ok=True)


def main() -> int:
    # The ONNX exporter prints emoji; a Windows console or a log opened with
    # the ANSI code page cannot encode them, and a print must not kill a run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "local" / "nexa.db")
    ap.add_argument("--job", required=True)
    ap.add_argument("--scope", choices=["single", "both"], default=None,
                    help="only 'single' exists for the 60-class model (default: the job's own choice)")
    ap.add_argument("--base", type=Path, default=None,
                    help=f"checkpoint to fine-tune when no retrained version is active (default {SHIPPED.relative_to(ROOT)})")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--replay", type=int, default=6000, help="original training windows mixed in")
    ap.add_argument("--correction-weight", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="fine-tuning step size: small, so the model adjusts rather than relearns")
    ap.add_argument("--gate", choices=["auto", "dataset", "synthetic"], default="auto")
    ap.add_argument("--gate-limit", type=int, default=6000,
                    help="windows of the test split in the exam, a fixed sample (0 = all 120k: slow on a CPU)")
    ap.add_argument("--synthetic-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-drop", type=float, default=0.01,
                    help="largest recall drop allowed on a judged class (0.01 = 1 point)")
    a = ap.parse_args()

    db = LocalDB(a.db)
    job = db.get_job(a.job)
    if not job:
        print(f"no such job {a.job}", file=sys.stderr)
        return 2
    scope = a.scope or job.get("scope") or "single"
    db.update_job(a.job, status="running")
    t0 = time.time()
    try:
        if scope != "single":
            raise RuntimeError("There is no 60-class ensemble to retrain: only the single model "
                               "(scope 'single') exists since the switch to 60 classes.")
        import torch
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        from src.config import CFG
        from src.models.amc_cnn import AMC_CNN
        from src.train import compute_class_weights

        # The dashboard's decision line (web/build.py writes the same into web/thresholds.js).
        per_class = CFG.get("web_thresholds_per_class") or {}
        thr = np.asarray([float(per_class.get(c, CFG.get("web_threshold", 0.3))) for c in CLASSES])
        n_steps = 4
        step = iter(range(1, n_steps + 1))
        say(f"Retrain job {a.job[:8]} started by {job['started_by']}: \"{job['reason']}\""
            + (" (OVERRIDE: rules not all met)" if job["override"] else ""))

        say(f"[{next(step)}/{n_steps}] Collecting approved corrections…")
        Xc, yc, used, skipped, family_only = correction_windows(db)
        say(f"      {len(used)} corrections -> {len(Xc)} windows"
            + (f"; {len(skipped)} skipped (no usable raw IQ)" if skipped else "")
            + (f"; {len(family_only)} skipped (labelled only 'JAMMING', no jammer type)" if family_only else ""))
        if not len(Xc):
            raise RuntimeError("No approved corrections with usable raw IQ and a 60-class label: "
                               "approve some corrections first.")

        have_data = dataset_available()
        gate_kind = a.gate if a.gate != "auto" else ("dataset" if have_data else "synthetic")
        if gate_kind == "dataset" and not have_data:
            raise RuntimeError("--gate dataset needs data/combined/test")

        say(f"[{next(step)}/{n_steps}] Preparing replay data and the fixed exam…")
        rng = np.random.default_rng(a.seed)
        if have_data:
            Xtr, ytr, Xte, yte = dataset_split()
        if gate_kind == "dataset":
            n = len(Xte) if not a.gate_limit else min(a.gate_limit, len(Xte))
            idx = np.random.default_rng(12345).choice(len(Xte), n, replace=False)   # fixed: same exam every run
            Xg, yg = take(Xte, yte, idx)
            gate_set = (f"held-out TEST split of data/combined ({len(idx):,} of {len(Xte):,} windows, "
                        "fixed sample, never trained on)")
        else:
            Xg, yg = synthetic_set(a.synthetic_per_class, seed=12345)
            why = "chosen for this run" if have_data else "data/combined not found on this machine"
            gate_set = (f"SYNTHETIC exam ({len(Xg)} judged-class + noise windows, fixed seed; {why}): "
                        "numbers are NOT comparable to the official scorecard")
        say(f"      exam: {gate_set}")

        if have_data and a.replay:
            pick = rng.choice(len(Xtr), size=min(a.replay, len(Xtr)), replace=False)
            Xr, yr = take(Xtr, ytr, pick)
            replay_src = "data/combined/train"
        elif a.replay:
            Xr, yr = synthetic_set(max(a.replay // 4, 1), seed=a.seed)
            replay_src = "SYNTHETIC (data/combined not found on this machine)"
        else:
            Xr, yr = np.zeros((0, 2, WIN), np.float32), np.zeros((0, len(CLASSES)), np.float32)
            replay_src = "none"
        # The SAME loss the model was trained with (scripts/train_week1.py):
        # per-class pos_weight from the training labels, NOISE_FLOOR damped.
        # A plain BCE moves every class's probability scale.
        y_weights = np.asarray(ytr) if have_data else np.concatenate([yr, yc])
        dampen = {CLASSES.index(c): m for c, m in CFG.get("week1", {}).get("class_weight_multipliers", {}).items()}
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(y_weights, len(CLASSES), dampen=dampen))

        def load(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            meta = ck["meta"]
            if list(meta["classes"]) != list(CLASSES):
                raise RuntimeError(f"{path.name} is not a 60-class checkpoint in this class order")
            m = AMC_CNN(num_classes=len(CLASSES), input_len=meta["input_len"],
                        stft_n_fft=meta["stft_n_fft"], **meta.get("model_flags", {}))
            m.load_state_dict(ck["state_dict"])
            return m.eval(), meta

        def rel(path):
            return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()

        active = db.active_model("single")
        if active and (active.get("metrics") or {}).get("n_classes") != len(CLASSES):
            active = None                    # an 8-class version from before the switch: not a base
        base_pt = (db.path.parent / Path(active["path"]).with_suffix(".pt")) if active else (a.base or SHIPPED)
        if not base_pt.exists():
            raise RuntimeError(f"{rel(base_pt)} not found: the 60-class checkpoint is git-ignored, "
                               "copy it there (docs/DATA_VERSIONS.md)")
        say(f"[{next(step)}/{n_steps}] Fine-tuning from "
            f"{'retrained ' + active['version'] if active else rel(base_pt)}…")
        t_train = time.time()
        torch.manual_seed(a.seed)
        model, base_meta = load(base_pt)
        Xt, yt = np.concatenate([Xr, Xc]), np.concatenate([yr, yc])
        w = np.concatenate([np.ones(len(Xr)), np.full(len(Xc), a.correction_weight)])
        loader = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(yt)), batch_size=64,
                            sampler=WeightedRandomSampler(torch.tensor(w), num_samples=len(Xt), replacement=True))
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            model.train()
            # BatchNorm keeps its learned statistics: a few hundred
            # fine-tuning batches would otherwise overwrite what ~1M windows
            # established.
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            say(f"      epoch {ep + 1}/{a.epochs}: loss {total / len(Xt):.4f}"
                f" ({len(Xr)} replay from {replay_src} + {len(Xc)} correction windows)")
        model.eval()

        say(f"[{next(step)}/{n_steps}] The exam: old vs new on the same windows…")
        old, _ = load(base_pt)
        before, after = score(old, Xg, yg, thr), score(model, Xg, yg, thr)
        passed, rules = gate(before, after, a.max_drop)
        for r in rules:
            say(f"      {'PASS' if r['met'] else 'FAIL'}  {r['text']}")
        fit = {"windows": len(Xc), "before": exact_fit(old, Xc, yc, thr), "after": exact_fit(model, Xc, yc, thr)}
        say(f"      corrections now matched exactly: {fit['before']:.1%} -> {fit['after']:.1%}")
        say(f"      Single model GATE {'PASSED' if passed else 'FAILED'}")
        g = {"set": gate_set, "kind": gate_kind, "passed": passed, "rules": rules,
             "before": before, "after": after}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version = f"ft-{stamp}"
        out = db.path.parent / "models" / version
        out.mkdir(parents=True, exist_ok=True)
        meta = {**base_meta, "fine_tuned": {"from": rel(base_pt), "job_id": a.job, "corrections": used,
                                            "replay_windows": int(len(Xr)), "epochs": a.epochs, "lr": a.lr}}
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "best_model.pt")
        say("      exporting to ONNX for the dashboard and checking it against PyTorch…")
        export_onnx_file(out / "best_model.pt", out / "best_model.onnx")
        metrics = {"gate": g, "corrections_fit": fit, "base": rel(base_pt), "n_classes": len(CLASSES),
                   "train": {"epochs": a.epochs, "lr": a.lr, "replay_windows": int(len(Xr)),
                             "correction_windows": int(len(Xc)), "corrections_used": used,
                             "corrections_family_only": family_only,
                             "seconds": round(time.time() - t_train, 1)}}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        verdict = "gate passed: waiting for approval" if passed else "gate FAILED: cannot be activated"
        candidates = [{"version": version, "kind": "single",
                       "path": (out / "best_model.onnx").relative_to(db.path.parent).as_posix(),
                       "source": {"job_id": a.job, "corrections": used, "replay_windows": int(len(Xr)),
                                  "base": metrics["base"]},
                       "metrics": metrics, "notes": verdict}]
        seconds = round(time.time() - t0, 1)
        db.finish_job(a.job, True, {"seconds": seconds, "scope": scope, "gate_passed": {"single": passed}},
                      candidates=candidates)
        say(f"      candidate {version}: " + ("PASSED, waiting for approval on the Model page" if passed
                                               else "FAILED its gate, cannot be activated"))
        say(f"Done in {seconds} s.")
        return 0
    except Exception as e:                                           # noqa: BLE001
        traceback.print_exc()
        db.finish_job(a.job, False, {"error": str(e)})
        say(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
