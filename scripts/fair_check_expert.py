"""
Fair check for the expert features, without retraining the CNN: do they add
anything the trained model does not already know, across all 60 classes?

Takes a trained week-1 member (best.pt), runs it over val and test, and
keeps the vector it hands to fc1 (its learned summary of the window). Then
trains two small identical heads on val, scores both on test:

  A  CNN summary alone                    (control: same head, no features)
  B  CNN summary + expert features        (src/expert_features.py, 104 values)

Same data, same head, same training -- the only difference is the features.
B beating A means the features carry information the CNN missed; B equal to
A means the CNN already had it and retraining with them is not worth 2 h.

Writes <out>/fair_check.json and fair_check.png (overall, per family, per
class at >= 10 dB).

    python scripts/fair_check_expert.py --ckpt results/2026-11-04_nfft64_ep15_seed1/best.pt
In Colab (GPU, ~5 min):
    !python scripts/fair_check_expert.py --ckpt {CKPT_DIR}/best.pt --data /content/combined --out {CKPT_DIR}/fair_check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASS_GROUPS, DATASET_CLASSES, REPO_ROOT  # noqa: E402
from src.models.amc_cnn import AMC_CNN, ExpertFeatures  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = len(DATASET_CLASSES)
FAMILY = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
FAMILY["NOISE_FLOOR"] = "NOISE"


def load(data, split, limit):
    d = Path(data) / split
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(limit, len(X)) if limit else len(X)
    return X[:n], np.load(d / "y.npy", mmap_mode="r")[:n].astype(bool), np.load(d / "snr_labels.npy")[:n]


def summarize(model, expert, X, batch=512):
    """(CNN summary before fc1, expert features), both float32 (N, *)."""
    grabbed = {}
    hook = model.fc1.register_forward_hook(lambda m, inp, out: grabbed.__setitem__("x", inp[0]))
    emb, feat = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[s:s + batch])).to(DEVICE).float()
            model(xb)
            emb.append(grabbed["x"].cpu().numpy())
            feat.append(expert(xb).cpu().numpy())
            if s // batch % 50 == 0:
                print(f"    {s + len(xb)}/{len(X)}  {time.time() - t0:.0f}s", flush=True)
    hook.remove()
    return np.concatenate(emb), np.concatenate(feat)


def train_head(Xtr, ytr, Xva, yva, seed, epochs=25):
    """Small MLP head, BCE like the trainer, best epoch on a held-out part of val."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda a: torch.from_numpy((a - mu) / sd).float().to(DEVICE)  # noqa: E731
    xt, yt, xv = norm(Xtr), torch.from_numpy(ytr).float().to(DEVICE), norm(Xva)
    head = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, C)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = -1, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(xt), device=DEVICE)
        for s in range(0, len(xt), 512):
            i = perm[s:s + 512]
            opt.zero_grad()
            lossf(head(xt[i]), yt[i]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = top1_hits(head(xv).cpu().numpy(), yva).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, norm


def top1_hits(logits, y):
    return y[np.arange(len(y)), logits.argmax(1)]


def scores(hits, y, snr, high=10):
    single = y.sum(1) == 1
    true = y.argmax(1)
    fam = np.array([FAMILY[DATASET_CLASSES[t]] for t in true])
    out = {"top1": float(hits.mean()), "top1_high": float(hits[snr >= high].mean()),
           "family_high": {}, "class_high": {}}
    for g in sorted(set(FAMILY.values())):
        m = single & (fam == g) & (snr >= high)
        out["family_high"][g] = float(hits[m].mean()) if m.any() else float("nan")
    for i, c in enumerate(DATASET_CLASSES):
        m = single & (true == i) & (snr >= high)
        out["class_high"][c] = float(hits[m].mean()) if m.any() else float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a trained week-1 best.pt")
    p.add_argument("--data", default=str(REPO_ROOT / CFG["week1"]["combined_data"]))
    p.add_argument("--out", default=None, help="default: <ckpt folder>/fair_check")
    p.add_argument("--limit", type=int, default=0, help="first N windows of val and of test (quick check)")
    p.add_argument("--seeds", type=int, default=2, help="heads trained per variant (results averaged)")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    meta = ck["meta"]
    if meta.get("model_flags", {}).get("expert_features"):
        sys.exit("This best.pt already has the expert features built in; the fair check needs a model trained without them.")
    model = AMC_CNN(num_classes=C, input_len=meta["input_len"], stft_n_fft=meta["stft_n_fft"],
                    **meta.get("model_flags", {})).to(DEVICE).eval()
    model.load_state_dict(ck["state_dict"])
    expert = ExpertFeatures(meta["input_len"]).to(DEVICE).eval()
    out = Path(args.out or Path(args.ckpt).parent / "fair_check")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {DEVICE}; model from epoch {meta['epoch']} (seed {meta['seed']})")

    data = {}
    for split in ("val", "test"):
        X, y, snr = load(args.data, split, args.limit)
        print(f"  {split}: {len(X)} windows")
        emb, feat = summarize(model, expert, X)
        data[split] = (emb, feat, y, snr)

    emb_v, feat_v, y_v, _ = data["val"]
    emb_t, feat_t, y_t, snr_t = data["test"]
    # 85% of val trains the heads, 15% picks their best epoch; test only scores.
    cut = int(0.85 * len(emb_v))
    variants = {"A: CNN only": (emb_v, emb_t),
                "B: CNN + expert": (np.hstack([emb_v, feat_v]), np.hstack([emb_t, feat_t]))}
    res = {}
    for name, (tr, te) in variants.items():
        runs = []
        for seed in range(args.seeds):
            head, norm = train_head(tr[:cut], y_v[:cut], tr[cut:], y_v[cut:], seed)
            with torch.no_grad():
                hits = top1_hits(head(norm(te)).cpu().numpy(), y_t)
            runs.append(scores(hits, y_t, snr_t))
        res[name] = {
            "top1": float(np.mean([r["top1"] for r in runs])),
            "top1_high": float(np.mean([r["top1_high"] for r in runs])),
            "family_high": {g: float(np.nanmean([r["family_high"][g] for r in runs])) for g in runs[0]["family_high"]},
            "class_high": {c: float(np.nanmean([r["class_high"][c] for r in runs])) for c in DATASET_CLASSES},
            "spread_top1": float(np.ptp([r["top1"] for r in runs])),
        }
        print(f"{name:>16}: top-1 {res[name]['top1']:.3f}   at >=10 dB {res[name]['top1_high']:.3f}"
              f"   (seed spread {res[name]['spread_top1']:.3f})", flush=True)
    with torch.no_grad():
        own = top1_hits(model.fc2(torch.relu(model.fc1(torch.from_numpy(emb_t).to(DEVICE)))).cpu().numpy(), y_t)
    res["CNN as trained"] = scores(own, y_t, snr_t)
    print(f"  CNN as trained: top-1 {res['CNN as trained']['top1']:.3f}   at >=10 dB {res['CNN as trained']['top1_high']:.3f}")

    a, b = res["A: CNN only"], res["B: CNN + expert"]
    print("\nPer family at >= 10 dB (A -> B):")
    for g in a["family_high"]:
        print(f"  {g:>13}  {a['family_high'][g]:.3f} -> {b['family_high'][g]:.3f}  ({b['family_high'][g] - a['family_high'][g]:+.3f})")
    delta = sorted(((b["class_high"][c] - a["class_high"][c], c) for c in DATASET_CLASSES), reverse=True)
    print("\nBiggest gains:", ", ".join(f"{c} {d:+.2f}" for d, c in delta[:8]))
    print("Biggest losses:", ", ".join(f"{c} {d:+.2f}" for d, c in delta[-5:]))
    (out / "fair_check.json").write_text(json.dumps({"ckpt": str(args.ckpt), "limit": args.limit, **res}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(a["family_high"])
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2.4]})
    x = np.arange(len(fams))
    ax[0].bar(x - 0.2, [a["family_high"][g] for g in fams], 0.4, label="A: CNN only", color="#9aa5b1")
    ax[0].bar(x + 0.2, [b["family_high"][g] for g in fams], 0.4, label="B: + expert features", color="#2f7ed8")
    ax[0].set_xticks(x); ax[0].set_xticklabels(fams, rotation=60, fontsize=8); ax[0].set_ylim(0, 1)
    ax[0].set_title("Top-1 per family, test, >= 10 dB"); ax[0].legend(fontsize=8)
    d = [b["class_high"][c] - a["class_high"][c] for c in DATASET_CLASSES]
    ax[1].bar(range(C), d, color=["#2f7ed8" if v >= 0 else "#d9534f" for v in d])
    ax[1].set_xticks(range(C)); ax[1].set_xticklabels(DATASET_CLASSES, rotation=90, fontsize=6)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_title("Change per class (B - A), test, >= 10 dB")
    fig.suptitle(f"Fair check: top-1 {a['top1']:.3f} -> {b['top1']:.3f} (all SNR), "
                 f"{a['top1_high']:.3f} -> {b['top1_high']:.3f} (>= 10 dB)")
    fig.tight_layout(); fig.savefig(out / "fair_check.png", dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
