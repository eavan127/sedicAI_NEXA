"""Export held-out test windows as drop-in IQ files for the OMNI console.

Produces raw interleaved-float32 captures that can be uploaded through RF
Replay's "Upload raw IQ" control, plus a manifest naming the true labels. The
point is to let someone verify the system on data it was never trained on,
using the same upload path any real recording would take.

Every window comes from the TEST split -- never trained on, and never used to
select thresholds either (those were chosen on validation). It is the only
portion of the dataset that has been kept back for exactly this.

Each capture concatenates consecutive windows of one class so the file is
long enough to exercise the sliding-window timeline rather than collapsing to
a single window. Consecutive dataset windows are independent captures, so
those joins are real discontinuities -- fine for verifying that the console
reads a file and classifies it, not a basis for quoting a performance number.
For that, use `python -m src.evaluate`, which scores the whole split properly.

Usage:
    python scripts/export_verification_pack.py --out verification_pack
    python scripts/export_verification_pack.py --out pack --snr -10 --windows 40
    python scripts/export_verification_pack.py --out pack --format sigmf
"""
import argparse
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402


def _spread_over_snr(indices, snr_labels, n, rng):
    """Pick n indices spread evenly over the SNR bins present.

    A verification capture drawn from one bin says only how the system does at
    that bin. Spreading makes a single upload exercise the whole sweep, which
    is what someone dropping in one file actually wants to see.
    """
    bins = sorted({float(snr_labels[i]) for i in indices})
    per = max(n // max(len(bins), 1), 1)
    out = []
    for b in bins:
        pool = indices[snr_labels[indices] == b]
        if len(pool):
            out.extend(pool[:per])
    return np.array(out[:n]) if out else indices[:n]


def write_iq(path, iq):
    """Interleaved float32 I,Q,I,Q,... -- the format RF Replay expects."""
    raw = np.empty(len(iq) * 2, dtype=np.float32)
    raw[0::2] = np.real(iq)
    raw[1::2] = np.imag(iq)
    raw.tofile(path)
    return raw.nbytes


def write_sigmf(base, iq, segments, description):
    """SigMF recording: base.sigmf-data + base.sigmf-meta, plus base.sigmf
    (the same two files in one tar, the spec's single-file form).

    cf32_le is byte-for-byte the interleaved float32 write_iq produces; the
    meta adds what a bare .f32 cannot say -- sample rate, datatype, and one
    annotation per true-class segment, which the console draws as TRUTH.
    `segments` are (class, start_sample, sample_count).
    """
    base = Path(base)
    data = base.with_name(base.name + ".sigmf-data")
    meta = base.with_name(base.name + ".sigmf-meta")
    size = write_iq(data, iq)
    meta.write_text(json.dumps({
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(CFG["signal"]["fs"]),
            "core:version": "1.2.0",
            "core:description": description,
            "core:author": "NEXA (SEDIC 2026)",
            "core:recorder": "export_verification_pack.py (held-out test split)",
            "core:license": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        },
        "captures": [{
            "core:sample_start": 0,
            "core:datetime": datetime.now(timezone.utc)
                                     .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }],
        "annotations": [
            {"core:sample_start": int(s), "core:sample_count": int(n),
             "core:label": cls}
            for cls, s, n in segments
        ],
    }, indent=2))
    archive = base.with_name(base.name + ".sigmf")
    with tarfile.open(archive, "w") as tar:
        # Spec: the archive holds one directory named after the recording.
        for f in (meta, data):
            tar.add(f, arcname=f"{base.name}/{f.name}")
    return size


def main(out_dir, n_windows, snr_filter, seed, fmt="f32"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    _, _, test = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"],
                                   d["seed"])
    print(f"held-out test split: {len(test):,} windows\n")

    manifest = []
    for cls in CLASSES:
        j = CLASSES.index(cls)
        sel = test[(y[test][:, j] > 0.5) & (y[test].sum(axis=1) == 1)]
        if snr_filter is not None:
            sel = sel[snr_labels[sel] == snr_filter]
        if len(sel) == 0:
            print(f"  {cls:<12} no standalone windows match -- skipped")
            continue

        # Spread across SNR bins. The test indices are ordered by bin, so
        # taking the first N landed every capture at -10 dB -- the hardest
        # case, and a misleading default for a verification file.
        take = _spread_over_snr(sel, snr_labels, n_windows, rng)
        iq = np.concatenate([X[i][0] + 1j * X[i][1] for i in take])
        tag = cls.lower() + (f"_{int(snr_filter):+d}dB" if snr_filter is not None else "")
        path = out / f"{tag}.f32"
        if fmt in ("f32", "both"):
            size = write_iq(path, iq)
        if fmt in ("sigmf", "both"):
            size = write_sigmf(out / tag, iq, [(cls, 0, len(iq))],
                               f"{cls}: {len(take)} held-out test windows")
            if fmt == "sigmf":
                path = out / f"{tag}.sigmf"

        snrs = sorted({float(snr_labels[i]) for i in take})
        manifest.append({
            "file": path.name, "true_class": cls, "windows": int(len(take)),
            "samples": int(len(iq)), "snr_db": snrs, "bytes": size,
        })
        print(f"  {cls:<12} {len(take):>3} windows -> {path.name}"
              f"  ({size / 1024:.0f} KB, SNR {snrs})")

    # A mixed capture: several classes back to back, to exercise the timeline
    # with more than one class in a single upload.
    parts, spans, pos = [], [], 0
    for cls in ["NOISE_FLOOR", "LFM_RADAR", "FHSS", "JAMMING", "QPSK"]:
        j = CLASSES.index(cls)
        sel = test[(y[test][:, j] > 0.5) & (y[test].sum(axis=1) == 1)]
        if not len(sel):
            continue
        take = _spread_over_snr(sel, snr_labels, max(n_windows // 2, 4), rng)
        seg = np.concatenate([X[i][0] + 1j * X[i][1] for i in take])
        parts.append(seg)
        spans.append({"class": cls,
                      "start_sample": pos, "end_sample": pos + len(seg)})
        pos += len(seg)
    if parts:
        iq = np.concatenate(parts)
        path = out / "mixed_sequence.f32"
        if fmt in ("f32", "both"):
            size = write_iq(path, iq)
        if fmt in ("sigmf", "both"):
            size = write_sigmf(
                out / "mixed_sequence", iq,
                [(s["class"], s["start_sample"], s["end_sample"] - s["start_sample"])
                 for s in spans],
                "Held-out test windows, several classes back to back")
            if fmt == "sigmf":
                path = out / "mixed_sequence.sigmf"
        manifest.append({"file": path.name, "true_class": "SEQUENCE",
                          "windows": int(pos // CFG["signal"]["window_len"]),
                          "samples": int(pos), "bytes": size,
                          "segments": spans})
        print(f"\n  {'mixed':<12} {len(spans)} classes  -> {path.name} "
              f"({size / 1024:.0f} KB)")

    (out / "manifest.json").write_text(json.dumps({
        "source": "held-out test split (never trained on, never used for "
                   "threshold selection)",
        "format": {"f32": "raw interleaved float32 I,Q,I,Q,...",
                   "sigmf": "SigMF cf32_le (.sigmf-meta + .sigmf-data, "
                            "and the same pair as one .sigmf archive)",
                   "both": "both of the above"}[fmt],
        "sample_rate_hz": CFG["signal"]["fs"],
        "window_len": CFG["signal"]["window_len"],
        "caveat": "windows are independent captures concatenated back to "
                   "back, so each 512-sample boundary is a discontinuity. "
                   "Use for verifying the console reads and classifies an "
                   "uploaded file; use src/evaluate.py for performance "
                   "numbers.",
        "files": manifest,
    }, indent=2))
    print(f"\nmanifest -> {out / 'manifest.json'}")
    print("Upload a .f32, a .sigmf archive, or a .sigmf-meta + .sigmf-data "
          "pair through 'Select & Analyze Source File'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="verification_pack")
    p.add_argument("--windows", type=int, default=60,
                    help="windows per class capture (60 = ~9.6 ms)")
    p.add_argument("--snr", type=float, default=None,
                    help="restrict to one SNR bin, e.g. -10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["f32", "sigmf", "both"], default="f32",
                    help="raw .f32 (default), SigMF recordings, or both")
    a = p.parse_args()
    main(a.out, a.windows, a.snr, a.seed, a.format)
