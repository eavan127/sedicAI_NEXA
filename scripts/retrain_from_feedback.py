"""Retrain from human feedback: approved corrections -> candidate model.

Started by the local server when an expert clicks "Start retraining" on the
Model page (POST /api/retrain/start), or by hand:

    python scripts/retrain_from_feedback.py --job <job id>

What it does, in order (each step is printed, and the page shows the log):

  1. Collect every APPROVED correction and cut its raw IQ into 512-sample
     windows labelled with what the human said was there. Pending and
     rejected corrections are never used.
  2. Mix them with a REPLAY sample of the original training data (the
     training split of data/processed): fine-tuning on corrections alone
     makes a model forget what it knew ("catastrophic forgetting").
  3. Fine-tune the single model (the active retrained one, else the shipped
     results/best_model.pt) for a couple of epochs, corrections up-weighted.
  4. THE GATE: score the old and the new model on the same fixed exam -- the
     held-out TEST split, never trained on and never containing a correction.
     The new model passes only if no judged class (LFM_RADAR, FHSS, JAMMING)
     loses more than 1 point of recall, none falls below 80% unless the old
     one was already below it, and pure-noise false alarms do not rise by
     more than 0.5 points. Without data/processed (a teammate's machine), the
     exam is a fixed synthetic set of judged-class windows, and the report
     says so.
  5. Export the new model to ONNX (verified against PyTorch) and register it
     as a CANDIDATE. Nothing changes for operators until a second person
     approves it on the Model page.

Scope (chosen on the Model page, stored on the job):
  single  the single model only (about 4 minutes on a laptop CPU)
  both    the single model AND all 5 ensemble members (about 20 minutes).
          Each member gets its own replay sample and seed so the five stay
          different; the ensemble sits the exam as the averaged whole, exactly
          like the submission, and becomes its own candidate needing its own
          approval.

Nothing here overwrites a shipped file: every version is written to
data/local/models/<version>/ and kept forever, whether it passed or not. The
shipped best_model.pt and the submitted ensemble stay untouched in results/
and web/models/, so "use shipped" always restores them exactly. The calibrated
thresholds are kept, so old and new are judged at the same decision points.
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

from src.localdb import CLASSES, JUDGED, LocalDB  # noqa: E402
from src.report_export import _iq_payload  # noqa: E402

FS = 3_200_000
WIN = 512
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


def correction_windows(db: LocalDB, max_per_correction: int = 200):
    """Approved corrections -> (X, y, ids). Windows are cut every 256 samples
    inside the corrected span; a span shorter than one window gets one window
    centred on it."""
    from src.config import multi_hot
    from src.data.preprocess import preprocess_window

    X, y, used, skipped = [], [], [], []
    cache = {}
    for c in db.list_corrections(status="approved", limit=100000):
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
            X.append(preprocess_window(iq[s:s + WIN]))
            y.append(label)
        used.append(c["id"])
    if not X:
        return np.zeros((0, 2, WIN), np.float32), np.zeros((0, len(CLASSES)), np.float32), used, skipped
    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32), used, skipped


def dataset_available() -> bool:
    from src.config import CFG
    d = ROOT / CFG["paths"]["processed_data"]
    return all((d / f).exists() for f in ("X.npy", "y.npy", "snr_labels.npy"))


def dataset_split():
    from src.config import CFG
    from src.train import stratified_split
    d = ROOT / CFG["paths"]["processed_data"]
    X = np.load(d / "X.npy", mmap_mode="r")
    y = np.load(d / "y.npy")
    snr = np.load(d / "snr_labels.npy")
    ds = CFG["dataset"]
    train, _, test = stratified_split(y, snr, ds["val_frac"], ds["test_frac"], ds["seed"])
    return X, y, np.asarray(train), np.asarray(test)


def synthetic_set(n_per_class: int, seed: int):
    """A FIXED synthetic exam of judged-class + noise windows (same seed ->
    same windows), for machines without data/processed. Civilian classes need
    RadioML and are left out."""
    from src.config import multi_hot
    from src.data.preprocess import add_awgn, preprocess_window
    from src.generators.fhss import random_fhss_example
    from src.generators.jamming import random_jamming_example
    from src.generators.radar import generate_lfm_chirp_iq

    rng = np.random.default_rng(seed)
    dur = WIN / FS
    makers = {
        # a chirp filling the window: a pulse train can leave a 160 us window silent
        "LFM_RADAR": lambda: generate_lfm_chirp_iq(FS, dur, rng.uniform(2e5, 1.2e6)),
        "FHSS": lambda: random_fhss_example(FS, dur, rng),
        "JAMMING": lambda: random_jamming_example(FS, dur, rng),
        "NOISE_FLOOR": lambda: np.zeros(WIN, complex),
    }
    X, y = [], []
    for cls, make in makers.items():
        for i in range(n_per_class):
            out = make()
            sig = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=complex)[:WIN]
            sig = np.pad(sig, (0, WIN - len(sig)))
            snr = [-6, -2, 2, 6, 10][i % 5]
            iq = (add_awgn(sig, snr, rng) if cls != "NOISE_FLOOR"
                  else (rng.standard_normal(WIN) + 1j * rng.standard_normal(WIN)) / np.sqrt(2))
            X.append(preprocess_window(iq))
            y.append(multi_hot([cls]))
    return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32)


N_MEMBERS = 5


def predict(models, X, batch: int = 512) -> np.ndarray:
    """Sigmoid probabilities averaged over `models` -- one model, or the five
    ensemble members exactly as the submission averages them."""
    import torch
    models = models if isinstance(models, (list, tuple)) else [models]
    total = None
    with torch.no_grad():
        for m in models:
            out = [torch.sigmoid(m(torch.tensor(np.asarray(X[i:i + batch])))).numpy()
                   for i in range(0, len(X), batch)]
            p = np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
            total = p if total is None else total + p
    return total / len(models)


def score(models, X, y, thr) -> dict:
    p = predict(models, X) > thr
    rec = {}
    for j, c in enumerate(CLASSES):
        pos = y[:, j] > 0.5
        rec[c] = float((p[pos, j]).mean()) if pos.any() else None
    noise_only = (y[:, CLASSES.index("NOISE_FLOOR")] > 0.5) & (y.sum(axis=1) == 1)
    non_noise = [j for j, c in enumerate(CLASSES) if c != "NOISE_FLOOR"]
    fa = float(p[noise_only][:, non_noise].any(axis=1).mean()) if noise_only.any() else None
    return {"recall": rec, "noise_false_alarm": fa}


def exact_fit(models, X, y, thr) -> float | None:
    """Share of correction windows where the model now says exactly what the human said."""
    if not len(X):
        return None
    p = predict(models, X) > thr
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
                    help="single model only, or single + the 5 ensemble members (default: the job's own choice)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--replay", type=int, default=12000, help="original training windows mixed in, per model")
    ap.add_argument("--correction-weight", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="fine-tuning step size: small, so the model adjusts rather than relearns")
    ap.add_argument("--gate", choices=["auto", "dataset", "synthetic"], default="auto")
    ap.add_argument("--gate-limit", type=int, default=0, help="subsample the exam (0 = all)")
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
        import torch
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        from src.config import resolve_class_weight_multipliers, resolve_multilabel_thresholds
        from src.models.amc_cnn import AMC_CNN
        from src.train import compute_class_weights

        thr = np.asarray(resolve_multilabel_thresholds())
        n_steps = 5 if scope == "both" else 4
        step = iter(range(1, n_steps + 1))
        say(f"Retrain job {a.job[:8]} started by {job['started_by']}: \"{job['reason']}\""
            + (" (OVERRIDE: rules not all met)" if job["override"] else ""))
        say("Scope: " + ("single model + the 5-model ensemble (about 20 min on a laptop CPU)"
                         if scope == "both" else "single model (about 4 min on a laptop CPU)"))

        say(f"[{next(step)}/{n_steps}] Collecting approved corrections…")
        Xc, yc, used, skipped = correction_windows(db)
        say(f"      {len(used)} corrections -> {len(Xc)} windows"
            + (f"; {len(skipped)} skipped (no usable raw IQ)" if skipped else ""))
        if not len(Xc):
            raise RuntimeError("No approved corrections with usable raw IQ: approve some corrections first.")

        have_data = dataset_available()
        gate_kind = a.gate if a.gate != "auto" else ("dataset" if have_data else "synthetic")
        if gate_kind == "dataset" and not have_data:
            raise RuntimeError("--gate dataset needs data/processed/X.npy")

        say(f"[{next(step)}/{n_steps}] Preparing replay data and the fixed exam…")
        rng = np.random.default_rng(a.seed)
        if have_data:
            X, y, train_idx, test_idx = dataset_split()
            y_all = np.load(ROOT / "data" / "processed" / "y.npy")
        else:
            X = y = train_idx = test_idx = None
            y_all = yc
        if gate_kind == "dataset":
            idx = test_idx if not a.gate_limit else np.sort(rng.choice(test_idx, min(a.gate_limit, len(test_idx)), replace=False))
            Xg, yg = np.asarray(X[idx]), y[idx]
            gate_set = f"held-out TEST split of data/processed ({len(idx):,} windows, never trained on)"
        else:
            Xg, yg = synthetic_set(a.synthetic_per_class, seed=12345)
            why = "chosen for this run" if have_data else "data/processed not found on this machine"
            gate_set = (f"SYNTHETIC exam ({len(Xg)} judged-class + noise windows, fixed seed; {why}): "
                        "numbers are NOT comparable to the official scorecard")
        say(f"      exam: {gate_set}")
        # The SAME loss the models were trained with (src/train.py): per-class
        # pos_weight from the full label set. A plain BCE moves every class's
        # probability scale, and the calibrated thresholds stop fitting.
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(
            y_all, len(CLASSES), dampen=resolve_class_weight_multipliers()))

        def replay(seed):
            """Each model gets its OWN replay sample: five ensemble members
            fine-tuned on identical data would drift together, and the
            ensemble helps only because its members make different mistakes."""
            r = np.random.default_rng(seed)
            if have_data and a.replay:
                pick = np.sort(r.choice(train_idx, size=min(a.replay, len(train_idx)), replace=False))
                return np.asarray(X[pick]), y[pick].astype(np.float32), "original training split"
            if a.replay:
                Xs, ys = synthetic_set(max(a.replay // 4, 1), seed=seed)
                return Xs, ys, "SYNTHETIC (data/processed not found on this machine)"
            return np.zeros((0, 2, WIN), np.float32), np.zeros((0, len(CLASSES)), np.float32), "none"

        def load(path):
            m = AMC_CNN(num_classes=len(CLASSES), input_len=WIN)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            return m.eval()

        def fine_tune(base_path, seed, label):
            torch.manual_seed(seed)
            Xr, yr, src = replay(seed)
            Xt, yt = np.concatenate([Xr, Xc]), np.concatenate([yr, yc])
            w = np.concatenate([np.ones(len(Xr)), np.full(len(Xc), a.correction_weight)])
            model = load(base_path)
            loader = DataLoader(TensorDataset(torch.tensor(Xt), torch.tensor(yt)), batch_size=64,
                                sampler=WeightedRandomSampler(torch.tensor(w), num_samples=len(Xt), replacement=True))
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            for ep in range(a.epochs):
                model.train()
                # BatchNorm keeps its learned statistics: a few hundred
                # fine-tuning batches would otherwise overwrite what 100k
                # windows established.
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
                say(f"      {label} epoch {ep + 1}/{a.epochs}: loss {total / len(Xt):.4f}"
                    f" ({len(Xr)} replay from {src} + {len(Xc)} correction windows)")
            return model.eval(), len(Xr)

        def judge(base_models, new_models, name):
            before, after = score(base_models, Xg, yg, thr), score(new_models, Xg, yg, thr)
            passed, rules = gate(before, after, a.max_drop)
            for r in rules:
                say(f"      {'PASS' if r['met'] else 'FAIL'}  {r['text']}")
            fit = {"windows": len(Xc), "before": exact_fit(base_models, Xc, yc, thr),
                   "after": exact_fit(new_models, Xc, yc, thr)}
            say(f"      corrections now matched exactly: {fit['before']:.1%} -> {fit['after']:.1%}")
            say(f"      {name} GATE {'PASSED' if passed else 'FAILED'}")
            return {"set": gate_set, "kind": gate_kind, "passed": passed, "rules": rules,
                    "before": before, "after": after}, fit

        def rel(path):
            return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()

        def verdict(g):
            return "gate passed: waiting for approval" if g["passed"] else "gate FAILED: cannot be activated"

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        candidates = []

        # ---- the single model -------------------------------------------------
        active = db.active_model("single")
        base_pt = (db.path.parent / Path(active["path"]).with_suffix(".pt")) if active \
            else ROOT / "results" / "best_model.pt"
        say(f"[{next(step)}/{n_steps}] Single model: fine-tuning from "
            f"{'retrained ' + active['version'] if active else 'shipped results/best_model.pt'}…")
        t_single = time.time()
        single, n_replay = fine_tune(base_pt, a.seed, "single")
        g, fit = judge(load(base_pt), single, "Single model")
        version = f"ft-{stamp}"
        out = db.path.parent / "models" / version
        out.mkdir(parents=True, exist_ok=True)
        torch.save(single.state_dict(), out / "best_model.pt")
        export_onnx_file(out / "best_model.pt", out / "best_model.onnx")
        metrics = {"gate": g, "corrections_fit": fit, "base": rel(base_pt),
                   "train": {"epochs": a.epochs, "lr": a.lr, "replay_windows": n_replay,
                             "correction_windows": int(len(Xc)), "corrections_used": used,
                             "seconds": round(time.time() - t_single, 1)}}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        candidates.append({"version": version, "kind": "single",
                           "path": (out / "best_model.onnx").relative_to(db.path.parent).as_posix(),
                           "source": {"job_id": a.job, "corrections": used, "replay_windows": n_replay,
                                      "base": metrics["base"]},
                           "metrics": metrics, "notes": verdict(g)})
        say(f"      single model done in {metrics['train']['seconds']} s -> candidate {version}")

        # ---- the 5-model ensemble ----------------------------------------------
        if scope == "both":
            active_ens = db.active_model("ensemble")
            base_dir = (db.path.parent / active_ens["path"]) if active_ens else ROOT / "results"
            base_paths = [base_dir / f"ensemble_{i}.pt" for i in range(N_MEMBERS)]
            missing = [p.name for p in base_paths if not p.exists()]
            if missing:
                raise RuntimeError(f"Ensemble checkpoints missing in {rel(base_dir)}: {', '.join(missing)}")
            say(f"[{next(step)}/{n_steps}] Ensemble: fine-tuning all {N_MEMBERS} members from "
                + (f"retrained {active_ens['version']}" if active_ens
                   else "the shipped submission (results/ensemble_*.pt)") + "…")
            t_ens = time.time()
            members, replays = [], []
            for i, bp in enumerate(base_paths):
                m, n = fine_tune(bp, a.seed + 101 * (i + 1), f"member {i + 1}/{N_MEMBERS}")
                members.append(m)
                replays.append(n)
            say("      exam: the averaged 5-model ensemble, old vs new, exactly like the submission")
            g, fit = judge([load(bp) for bp in base_paths], members, "Ensemble")
            version = f"ens-{stamp}"
            out = db.path.parent / "models" / version
            out.mkdir(parents=True, exist_ok=True)
            for i, m in enumerate(members):
                torch.save(m.state_dict(), out / f"ensemble_{i}.pt")
                export_onnx_file(out / f"ensemble_{i}.pt", out / f"ensemble_{i}.onnx")
            metrics = {"gate": g, "corrections_fit": fit, "base": rel(base_dir),
                       "train": {"epochs": a.epochs, "lr": a.lr, "replay_windows": int(sum(replays)),
                                 "correction_windows": int(len(Xc)), "corrections_used": used,
                                 "members": N_MEMBERS, "seconds": round(time.time() - t_ens, 1)}}
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            candidates.append({"version": version, "kind": "ensemble",
                               "path": out.relative_to(db.path.parent).as_posix(),
                               "source": {"job_id": a.job, "corrections": used,
                                          "replay_windows": int(sum(replays)), "base": metrics["base"]},
                               "metrics": metrics, "notes": verdict(g)})
            say(f"      ensemble done in {metrics['train']['seconds']} s -> candidate {version}")

        say(f"[{next(step)}/{n_steps}] Registering candidates…")
        seconds = round(time.time() - t0, 1)
        db.finish_job(a.job, True, {"seconds": seconds, "scope": scope,
                                    "gate_passed": {c["kind"]: c["metrics"]["gate"]["passed"] for c in candidates}},
                      candidates=candidates)
        for c in candidates:
            say(f"      {c['kind']:8s} {c['version']}: "
                + ("PASSED, waiting for approval on the Model page" if c["metrics"]["gate"]["passed"]
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
