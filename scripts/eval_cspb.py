"""External validation of the civilian classes against CSPB.ML.2018R2.

Our civilian classes come from RadioML alone, so scoring them on a RadioML test
split cannot distinguish "learned modulation" from "learned RadioML". CSPB is a
third-party dataset (Spooner, CSP Blog) covering the same four modulations with
randomised symbol rate, carrier offset and pulse-shaping roll-off -- the
parameters RadioML holds fixed. We never train on it.

Setup:
    data/raw/cspb/CSPB.ML.2018R2_1.zip          batch 1, 4000 signals (~920 MB)
    D:/cspb_batches/CSPB.ML.2018R2_N.zip        batches 2-28 (~920 MB each, ~25 GB)
    data/raw/cspb/signal_record_C_2023.txt      truth file, all 112000 rows

    Batch 1 downloads:  https://cyclostationary.blog/wp-content/uploads/2023/09/CSPB.ML_.2018R2_1.zip
    Truth file:         https://cyclostationary.blog/wp-content/uploads/2023/09/signal_record_C_2023.txt

Use the R2 revision, not the original CSPB.ML.2018: the original has an RNG flaw
that duplicated parameter sets across signals. Cite as "CSPB.ML.2018R2".
NOTE: the CSP Blog states a citation requirement but no explicit licence terms --
confirm permitted use before relying on this in a submission.

.tim format (verified against the files, not assumed):
    8-byte header = int32 n_components (2), int32 n_samples (32768)
    then n_samples interleaved complex float32

Truth columns:
    idx, modtype, base_symbol_period, carrier_offset, excess_bandwidth,
    U, D, inband_SNR_dB, noise_spectral_density (always 0)
    symbol rate = (1/base_symbol_period) * (D/U);  D == 0 means no resampling.

Usage:
    python scripts/eval_cspb.py
    python scripts/eval_cspb.py --checkpoint results/civjitter/best_model.pt
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.data.preprocess import preprocess_window  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

MAP = {"bpsk": "BPSK", "qpsk": "QPSK", "16qam": "16QAM", "64qam": "64QAM"}
PER_BATCH = 4000            # signal indices per batch zip
THREATS = ["LFM_RADAR", "FHSS", "JAMMING"]

# Batch 8's zip was uploaded missing signal_31986.tim. This is a known defect,
# reported in the CSPB.ML.2018R2 post's comments and acknowledged by the author,
# who posted the single file separately rather than re-upload a 1 GB zip. So
# batch 8 legitimately holds 3999 entries and must not be rejected for it.
# signal_31986 is a QPSK example, i.e. one of ours, so we splice it back in
# from the standalone zip when that file is available.
KNOWN_SHORT = {8: PER_BATCH - 1}
SUPPLEMENT_MEMBER = "signal_31986.tim"
SUPPLEMENT_IDX = 31986


def discover_batches(paths, expect_entries=PER_BATCH):
    """Map batch number -> zip path, from explicit files and/or directories.

    Batch N holds signal indices (N-1)*4000+1 .. N*4000, in Batch_Dir_N/.

    Zips that fail to open or hold the wrong number of entries are SKIPPED with
    a warning, not read. A directory may legitimately contain a download still
    in flight, and silently evaluating against a truncated batch would produce
    a plausible-looking but wrong number.
    """
    found, bad = {}, []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            continue
        candidates = sorted(p.glob("*.zip")) if p.is_dir() else [p]
        for c in candidates:
            m = re.search(r"2018R2[_.]?(\d+)\.zip$", c.name)
            if not m:
                continue
            num = int(m.group(1))
            want = KNOWN_SHORT.get(num, expect_entries)
            try:
                with zipfile.ZipFile(c) as z:
                    n = len(z.namelist())
            except Exception:
                bad.append(f"{c.name} (unreadable)")
                continue
            if n != want:
                bad.append(f"{c.name} ({n} entries, expected {want})")
                continue
            found[num] = c
    if bad:
        print("  ! skipping incomplete/corrupt zips: " + ", ".join(bad))
    return dict(sorted(found.items()))


def load_truth(path, batches):
    """Truth rows for our four civilian classes, restricted to available batches."""
    wanted = set()
    for b in batches:
        wanted.update(range((b - 1) * PER_BATCH + 1, b * PER_BATCH + 1))
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            idx = int(p[0])
            if idx not in wanted or p[1] not in MAP:
                continue
            base, U, D = float(p[2]), float(p[5]), float(p[6])
            rows.append(dict(idx=idx, mod=p[1], snr=float(p[7]),
                             batch=(idx - 1) // PER_BATCH + 1,
                             sps=1.0 / ((1.0 / base) * ((D / U) if D > 0 else 1.0)),
                             rolloff=float(p[4])))
    return rows


def _window(raw, window_len):
    a = np.frombuffer(raw[8:], dtype="<f4")
    iq = a[0::2] + 1j * a[1::2]
    start = (len(iq) - window_len) // 2          # centre, deterministic
    return preprocess_window(iq[start:start + window_len])


def build_eval_set(batches, rows, window_len, supplement=None):
    """Read one centred window per signal. Opens each zip once, in index order.

    Batch 8's zip is missing signal_31986.tim (see KNOWN_SHORT). If the
    separately-posted file is supplied it is spliced in; otherwise that one
    signal is skipped with a warning rather than aborting a multi-GB run.
    """
    extra = None
    if supplement:
        sp = Path(supplement)
        if not sp.is_absolute():
            sp = REPO_ROOT / sp
        if sp.exists():
            try:
                with zipfile.ZipFile(sp) as sz:
                    extra = sz.read(SUPPLEMENT_MEMBER)
            except Exception:
                extra = None

    by_batch = {}
    for r in rows:
        by_batch.setdefault(r["batch"], []).append(r)

    X, y, kept, missing, spliced = [], [], [], 0, 0
    for b, brows in sorted(by_batch.items()):
        with zipfile.ZipFile(batches[b]) as z:
            names = set(z.namelist())
            for r in brows:
                member = f"Batch_Dir_{b}/signal_{r['idx']}.tim"
                if member in names:
                    raw = z.read(member)
                elif r["idx"] == SUPPLEMENT_IDX and extra is not None:
                    raw, _ = extra, None
                    spliced += 1
                else:
                    missing += 1
                    continue
                X.append(_window(raw, window_len))
                y.append(CLASS_TO_IDX[MAP[r["mod"]]])
                kept.append(r)
    if spliced:
        print(f"  spliced in {spliced} signal from the standalone batch-8 file")
    if missing:
        print(f"  ! {missing} signal file(s) absent from the zips — skipped")
    return np.stack(X), np.array(y), kept


def main(checkpoint, zip_paths, truth_path, supplement=None):
    truth_path = Path(truth_path)
    if not truth_path.is_absolute():
        truth_path = REPO_ROOT / truth_path
    if not truth_path.exists():
        raise FileNotFoundError(f"{truth_path} missing — see this file's docstring for the URLs")

    batches = discover_batches(zip_paths)
    if not batches:
        raise FileNotFoundError(f"no CSPB batch zips found under {zip_paths}")

    window_len = CFG["signal"]["window_len"]
    rows = load_truth(truth_path, batches)

    print(f"CSPB.ML.2018R2 — {len(batches)} batch(es): "
          f"{', '.join(str(b) for b in batches)}")
    print(f"{len(rows)} signals in our 4 civilian classes")
    X, y, kept = build_eval_set(batches, rows, window_len, supplement)
    sps = np.array([r["sps"] for r in kept])
    snr = np.array([r["snr"] for r in kept])

    print(f"  samples/symbol : {sps.min():.1f} to {sps.max():.1f} (median {np.median(sps):.1f})")
    print("                   RadioML, which we train on, is fixed at ~8")
    print(f"  inband SNR dB  : {snr.min():.1f} to {snr.max():.1f}")
    print("  NOTE: CSPB inband SNR is defined differently from our add_awgn —")
    print("        the two SNR scales are not directly comparable.\n")

    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    model = AMC_CNN(num_classes=len(CLASSES), input_len=window_len)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(torch.tensor(X[i:i + 512])).argmax(1).numpy()
                               for i in range(0, len(X), 512)])
    print(f"checkpoint: {ckpt}\n")

    out = {"checkpoint": str(checkpoint), "batches": sorted(batches),
           "n": int(len(y)), "per_class": {}}
    print(f"{'class':<8}{'n':>6}{'recall':>9}   where predictions go")
    print("-" * 64)
    for name in ["BPSK", "QPSK", "16QAM", "64QAM"]:
        i = CLASS_TO_IDX[name]
        m = y == i
        cnt = np.bincount(pred[m], minlength=len(CLASSES))
        rec = float((pred[m] == i).mean())
        out["per_class"][name] = rec
        top = "  ".join(f"{CLASSES[j]}={cnt[j] / m.sum():.2f}"
                        for j in np.argsort(cnt)[::-1][:3])
        print(f"{name:<8}{m.sum():>6}{rec:>9.3f}   {top}")

    jam = CLASS_TO_IDX["JAMMING"]
    threat_idx = [CLASS_TO_IDX[c] for c in THREATS]
    out["accuracy"] = float((pred == y).mean())
    out["false_alarm_jamming"] = float((pred == jam).mean())
    out["false_alarm_any_threat"] = float(np.isin(pred, threat_idx).mean())

    print(f"\noverall accuracy          : {out['accuracy']:.3f}")
    print(f"civilian called JAMMING   : {out['false_alarm_jamming']:.4f}"
          "   <- compare with evals/scorecard.json false_alarm_rate")
    print(f"civilian called ANY threat: {out['false_alarm_any_threat']:.4f}")

    print("\n=== accuracy vs samples-per-symbol (the parameter RadioML fixes) ===")
    out["by_sps"] = {}
    for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 100)]:
        m = (sps >= lo) & (sps < hi)
        if m.sum() > 20:
            a = float((pred[m] == y[m]).mean())
            fa = float((pred[m] == jam).mean())
            out["by_sps"][f"{lo}-{hi}"] = {"n": int(m.sum()), "accuracy": a, "fa_jamming": fa}
            print(f"  {lo:>3}-{hi:<3} n={m.sum():>5}  acc={a:.3f}  jamming-FA={fa:.4f}")

    evals_dir = REPO_ROOT / CFG["paths"]["evals"]
    evals_dir.mkdir(parents=True, exist_ok=True)
    with open(evals_dir / "cspb_external.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {evals_dir / 'cspb_external.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/best_model.pt")
    p.add_argument("--zip", nargs="+",
                   default=["data/raw/cspb", "D:/cspb_batches"],
                   help="batch zip files and/or directories containing them")
    p.add_argument("--truth", default="data/raw/cspb/signal_record_C_2023.txt")
    p.add_argument("--supplement", default="D:/cspb_batches/signal_31986.tim_.zip",
                   help="standalone zip holding the file absent from batch 8")
    a = p.parse_args()
    main(a.checkpoint, a.zip, a.truth, a.supplement)
