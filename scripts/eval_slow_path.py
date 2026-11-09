"""
Before/after for the slow path: the zoom rules alone (src/pipeline/analog.py)
against the rules plus the slow-path model (src/pipeline/narrowband.refine),
on fresh synthetic zoomed channels (a seed the model never saw), per kind and
SNR. A rules answer counts as right when it names the same kind (the rules say
UNKNOWN for an empty band, and cannot name RTTY / PSK31 / AFSK at all).

    python scripts/eval_slow_path.py --n 60
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402
from src.pipeline import analog, narrowband as nb  # noqa: E402

SAME = {"NOISE": {"UNKNOWN"}, "CARRIER": {"CARRIER"}, "CW": {"CW"}, "RTTY": {"RTTY"}, "PSK31": {"PSK31"},
        "AFSK": {"AFSK"}, "AM": {"AM"}, "NBFM": {"NBFM"}, "USB": {"USB"}, "LSB": {"LSB"}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60, help="signals per kind")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "slow_path_eval.json"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for kind in nb.KINDS:
        for _ in range(a.n):
            y, snr = nb.make(kind, rng)
            rk, _, why = analog.classify(y, nb.FA)
            fk, _, _, _ = nb.refine(rk, why, y)
            rows.append((kind, snr, rk in SAME[kind], fk in SAME[kind]))
    out = {}
    print(f"{'kind':<9} {'rules':>7} {'rules+model':>12}   (clear >= 10 dB: rules / rules+model)")
    for kind in nb.KINDS:
        r = [x for x in rows if x[0] == kind]
        hi = [x for x in r if x[1] >= 10]
        out[kind] = {"rules": np.mean([x[2] for x in r]), "with_model": np.mean([x[3] for x in r]),
                     "rules_clear": np.mean([x[2] for x in hi]) if hi else None,
                     "with_model_clear": np.mean([x[3] for x in hi]) if hi else None, "n": len(r)}
        o = out[kind]
        print(f"{kind:<9} {o['rules']:7.0%} {o['with_model']:12.0%}   "
              f"({o['rules_clear']:.0%} / {o['with_model_clear']:.0%})")
    allr = np.mean([x[2] for x in rows])
    allm = np.mean([x[3] for x in rows])
    hi = [x for x in rows if x[1] >= 10]
    print(f"{'overall':<9} {allr:7.0%} {allm:12.0%}   ({np.mean([x[2] for x in hi]):.0%} / {np.mean([x[3] for x in hi]):.0%})")
    out["overall"] = {"rules": allr, "with_model": allm, "rules_clear": np.mean([x[2] for x in hi]),
                      "with_model_clear": np.mean([x[3] for x in hi]), "n": len(rows), "seed": a.seed}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
