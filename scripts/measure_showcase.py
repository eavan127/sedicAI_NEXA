"""
Per-class and per-family accuracy of the active classifier on fresh windows
from our own generators (impaired and placed anywhere in the band, like the
fixed data), for the exhibition slides.

For every class and SNR: N windows, scored one at a time and as a track vote
(mean log-odds of VOTE windows of the same emitter, as calibrate_week1 does).
Reports exact-class top-1 and right-family rates. Fresh generator windows are
cleaner than the test split (one emitter, no overlaps), so these numbers sit
above the test-set ones; the slides say so.

    python scripts/measure_showcase.py --n 64 --snr 10 15 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_specialists import GEN, make  # noqa: E402
from src.config import CLASS_GROUPS, DATASET_CLASSES as C  # noqa: E402

FAM = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
FAM["NOISE_FLOOR"] = "NOISE"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=64, help="windows per class and SNR (multiple of --vote)")
    p.add_argument("--vote", type=int, default=8)
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 15.0, 20.0])
    p.add_argument("--classifier", default="results/classifier/physics_aug_best.pt")
    p.add_argument("--out", default="results/accuracy_plan/showcase.json")
    p.add_argument("--merge-dense", action="store_true",
                   help="count QAM and APSK as one family, as the resolver answers them (\"QAM / APSK\")")
    a = p.parse_args()
    if a.merge_dense:
        for c, g in FAM.items():
            if g in ("QAM", "APSK"):
                FAM[c] = "QAM / APSK"
    from src.pipeline.classify import Classifier
    clf = Classifier([a.classifier])
    fam_of = np.array([FAM[c] for c in C])
    rng = np.random.default_rng(7)
    res = {}
    classes = [c for c in C if c in GEN]
    print("classes with a generator:", len(classes), "of", len(C), flush=True)
    for snr in a.snr:
        per = {}
        for c in classes:
            W = np.array([make(c, snr, rng, place=True) for _ in range(a.n)])
            P = np.clip(clf(W), 1e-4, 1 - 1e-4)
            k = C.index(c)
            top = P.argmax(1)
            L = np.log(P / (1 - P)).reshape(-1, a.vote, len(C)).mean(1)
            vtop = L.argmax(1)
            per[c] = {"family": FAM[c], "top1": float((top == k).mean()),
                      "family_right": float((fam_of[top] == FAM[c]).mean()),
                      "vote_top1": float((vtop == k).mean()),
                      "vote_family_right": float((fam_of[vtop] == FAM[c]).mean())}
        fams = {}
        for g in sorted(set(FAM[c] for c in classes)):
            cs = [c for c in classes if FAM[c] == g]
            fams[g] = {m: float(np.mean([per[c][m] for c in cs]))
                       for m in ("top1", "family_right", "vote_top1", "vote_family_right")}
        overall = {m: float(np.mean([per[c][m] for c in classes]))
                   for m in ("top1", "family_right", "vote_top1", "vote_family_right")}
        res[str(int(snr))] = {"overall": overall, "family": fams, "class": per}
        print(f"\n{snr:.0f} dB  overall: top-1 {overall['top1']:.3f}  family {overall['family_right']:.3f}  "
              f"vote top-1 {overall['vote_top1']:.3f}  vote family {overall['vote_family_right']:.3f}", flush=True)
        for g, v in fams.items():
            print(f"  {g:13s} top-1 {v['top1']:.2f}  family {v['family_right']:.2f}  "
                  f"vote {v['vote_top1']:.2f}  vote family {v['vote_family_right']:.2f}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"classifier": a.classifier, "n": a.n, "vote": a.vote, "snr": res}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
