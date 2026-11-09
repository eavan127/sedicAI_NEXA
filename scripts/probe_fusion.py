"""
Does a specialist that sees BOTH the physics features and the CNN's scores beat
each one alone? (The "team of specialists" design, measured before building it.)

Same windows for all three variants (placed + impaired, our generators):
  CNN         the 60-class CNN, argmax among the group's classes
  physics     gradient boosting on the physics features (probe_specialists.physics)
  fused       gradient boosting on physics features + the CNN's 60 log-odds

Groups where the CNN is not hurt by the FSK/PSK/QAM 0 Hz bug are used, so the
CNN is scored fairly on placed windows.

    python scripts/probe_fusion.py --n 200 --snr 15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_specialists import GROUPS, make, physics  # noqa: E402
from src.config import DATASET_CLASSES  # noqa: E402

FAIR = ["APSK vs QAM vs band noise", "Analog vs tones", "OFDM vs noise jamming", "Radar shapes", "Jammer kinds",
        "Pulse comms vs radar"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--snr", type=float, default=15.0)
    p.add_argument("--classifier", default="results/classifier/physics_aug_best.pt")
    a = p.parse_args()
    from src.pipeline.classify import Classifier
    clf = Classifier([a.classifier])
    rng = np.random.default_rng(21)
    res = {}
    print(f"{'group':28s} {'CNN':>6s} {'physics':>8s} {'fused':>6s}")
    for name in FAIR:
        classes = GROUPS[name]
        idx = [DATASET_CLASSES.index(c) for c in classes]
        W, y = [], []
        for k, c in enumerate(classes):
            for _ in range(a.n):
                W.append(make(c, a.snr, rng, place=True))
                y.append(k)
        W, y = np.array(W), np.array(y)
        P = np.clip(clf(W), 1e-4, 1 - 1e-4)
        L = np.log(P / (1 - P))
        F = np.nan_to_num(np.array([physics(w) for w in W]))
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=0, stratify=y)
        cnn = float((P[te][:, idx].argmax(1) == y[te]).mean())

        def fit(X):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(X[tr], y[tr])
            return float((m.predict(X[te]) == y[te]).mean())
        phys, fused = fit(F), fit(np.hstack([F, L]))
        res[name] = {"classes": classes, "cnn": cnn, "physics": phys, "fused": fused}
        print(f"{name:28s} {cnn:6.0%} {phys:8.0%} {fused:6.0%}", flush=True)
    out = Path(f"results/accuracy_plan/fusion_{int(a.snr)}dB.json")
    out.write_text(json.dumps({"snr_db": a.snr, "n_per_class": a.n, "groups": res}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
