"""
Calibrate a week-1 60-class model on VAL, then report the effect on TEST
(SED-39 item 2). No training. Three settings:

  temperature  one T for all classes: sigmoid(logit / T). Fitted by minimising
               BCE on val. Changes the confidences shown to the operator, never
               the answer (argmax is the same for any T > 0).
  noise_t      "no signal" rule: if the best class probability (after the
               temperature) is below noise_t, answer NOISE_FLOOR. Picked on val
               as the highest value where at most --max-lost of the signals at
               >= 10 dB get called noise.
  vote length  how many windows a track needs. Proxy: average the log-odds of
               N random test windows of the same class and SNR. They come from
               different generated bursts, so real tracks (one emitter, steady
               parameters) should do at least this well.

Writes <ckpt folder>/calibration.json and prints every table.

    python scripts/calibrate_week1.py --ckpt <best.pt> --data /content/combined
    python scripts/calibrate_week1.py --ckpt best.pt --data data/combined --cache scratch/logits
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIGH_DB = 10
VOTE_N = (1, 2, 4, 8, 16, 32, 64)
NOISE_GRID = np.round(np.arange(0.0, 0.501, 0.01), 2)


def load_model(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m = ck["meta"]
    model = AMC_CNN(num_classes=len(m["classes"]), input_len=m["input_len"],
                    stft_n_fft=m["stft_n_fft"], **m.get("model_flags", {})).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), m


@torch.no_grad()
def logits_of(model, X, batch=1024):
    out = np.zeros((len(X), model.fc2.out_features), np.float32)
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[s:s + batch], dtype=np.float32)).to(DEVICE)
        out[s:s + batch] = model(xb).cpu().numpy()
    return out


def split(data, name, model, cache):
    d = Path(data) / name
    y = np.load(d / "y.npy").astype(bool)
    snr = np.load(d / "snr_labels.npy")
    path = Path(cache) / f"logits_{name}.npy" if cache else None
    if path is not None and path.exists():
        z = np.load(path)
    else:
        z = logits_of(model, np.load(d / "X.npy", mmap_mode="r"))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, z)
    return z.astype(np.float64), y, snr


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def bce(z, y, t):
    p = np.clip(sigmoid(z / t), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean())


def fit_temperature(z, y):
    """Golden-section search for the T minimising BCE (convex enough in log T)."""
    lo, hi = np.log(0.25), np.log(4.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = bce(z, y, np.exp(a)), bce(z, y, np.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = bce(z, y, np.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = bce(z, y, np.exp(b))
    return float(np.exp((lo + hi) / 2))


def ece(p_top, hit, bins=15):
    """Expected calibration error of the top-1 confidence: how far 'says 80%'
    is from 'right 80% of the time', averaged over confidence bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_top, edges) - 1, 0, bins - 1)
    return float(sum(abs(hit[idx == b].mean() - p_top[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def noise_rule(p, y, snr, noise_idx, t):
    top = p.argmax(1)
    pred = np.where(p.max(1) < t, noise_idx, top)
    hit = y[np.arange(len(y)), pred]
    isn = y[:, noise_idx]
    sig, high = ~isn, ~isn & (snr >= HIGH_DB)
    called = pred == noise_idx
    return {"noise_found": float(hit[isn].mean()), "signal_top1": float(hit[sig].mean()),
            "signal_top1_high": float(hit[high].mean()),
            "signals_called_noise": float((called & sig & (top != noise_idx)).sum() / sig.sum()),
            "signals_called_noise_high": float((called & high & (top != noise_idx)).sum() / high.sum())}


def vote_curve(z, y, snr, noise_idx, rng, reps=4000):
    """Top-1 of the mean log-odds of N same-class, same-SNR windows (single-label
    windows only, so 'the class' is well defined)."""
    single = y.sum(1) == 1
    cls = y.argmax(1)
    groups = {}
    for i in np.flatnonzero(single):
        groups.setdefault((cls[i], float(snr[i])), []).append(i)
    keys = [k for k, v in groups.items() if len(v) >= max(VOTE_N)]
    out = {}
    for n in VOTE_N:
        hits = {"all": [], "high": []}
        for k in keys:
            members = np.array(groups[k])
            picks = rng.choice(members, size=(max(1, reps // len(keys)), n), replace=True)
            pred = z[picks].mean(1).argmax(1)
            h = (pred == k[0]).tolist()
            hits["all"] += h
            if k[1] >= HIGH_DB:
                hits["high"] += h
        out[n] = {"top1": float(np.mean(hits["all"])), "top1_high": float(np.mean(hits["high"]))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a week-1 best.pt")
    p.add_argument("--data", required=True, help="merged data folder with val/ and test/")
    p.add_argument("--cache", default=None, help="folder to keep the val/test logits in (reused next time)")
    p.add_argument("--max-lost", type=float, default=0.02,
                   help="noise_t: most signals at >= 10 dB the rule may call noise, on val")
    p.add_argument("--out", default=None, help="default: <ckpt folder>/calibration.json")
    args = p.parse_args()

    model, meta = load_model(args.ckpt)
    classes = list(meta["classes"])
    noise_idx = classes.index("NOISE_FLOOR")
    zv, yv, sv = split(args.data, "val", model, args.cache)
    zt, yt, st = split(args.data, "test", model, args.cache)

    # --- temperature
    t = fit_temperature(zv, yv)
    rows = {}
    for name, temp in (("raw", 1.0), ("calibrated", t)):
        pt = sigmoid(zt / temp)
        top = pt.argmax(1)
        hit = yt[np.arange(len(yt)), top]
        rows[name] = {"temperature": temp, "test_bce": bce(zt, yt, temp),
                      "test_ece_top1": ece(pt.max(1), hit), "mean_top1_confidence": float(pt.max(1).mean()),
                      "test_top1": float(hit.mean())}
    print(f"temperature fitted on val: T = {t:.3f}")
    print(f"{'':12}{'BCE':>8}{'ECE top-1':>11}{'mean conf':>11}{'top-1':>8}")
    for name, r in rows.items():
        print(f"{name:12}{r['test_bce']:8.4f}{r['test_ece_top1']:11.3f}{r['mean_top1_confidence']:11.3f}{r['test_top1']:8.3f}")

    # --- noise threshold, picked on val with the temperature applied
    pv, pt = sigmoid(zv / t), sigmoid(zt / t)
    ok = [g for g in NOISE_GRID if noise_rule(pv, yv, sv, noise_idx, g)["signals_called_noise_high"] <= args.max_lost]
    noise_t = float(max(ok)) if ok else 0.0
    print(f"\nnoise threshold picked on val: {noise_t:.2f} (at most {args.max_lost:.0%} of >= {HIGH_DB} dB signals called noise)")
    print(f"{'T':>5}{'noise found':>13}{'signal top-1':>14}{'>=10dB':>8}{'sig->noise':>12}{'>=10dB':>8}   (test)")
    noise_table = {}
    for g in sorted({0.0, 0.05, 0.1, 0.15, 0.2, 0.3, noise_t}):
        r = noise_rule(pt, yt, st, noise_idx, g)
        noise_table[f"{g:.2f}"] = r
        mark = "  <- picked" if g == noise_t else ""
        print(f"{g:5.2f}{r['noise_found']:13.3f}{r['signal_top1']:14.3f}{r['signal_top1_high']:8.3f}"
              f"{r['signals_called_noise']:12.3f}{r['signals_called_noise_high']:8.3f}{mark}")

    # --- vote length
    votes = vote_curve(zt, yt, st, noise_idx, np.random.default_rng(0))
    print("\nvote length (mean log-odds of N same-class, same-SNR test windows):")
    print(f"{'N':>4}{'top-1':>8}{'>=10dB':>8}")
    for n, r in votes.items():
        print(f"{n:4d}{r['top1']:8.3f}{r['top1_high']:8.3f}")

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "calibration.json"
    out.write_text(json.dumps({
        "ckpt": str(args.ckpt), "epoch": meta.get("epoch"), "combined_version": meta.get("combined_version"),
        "temperature": t, "noise_threshold": noise_t, "max_lost_high": args.max_lost,
        "temperature_effect": rows, "noise_table_test": noise_table,
        "vote_length_test": {str(k): v for k, v in votes.items()},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
