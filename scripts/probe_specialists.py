"""
Deep separability check: for every group of classes the 60-class CNN confuses,
can hand-built PHYSICS measurements tell the members apart, and better than
the CNN does?

Per group a "specialist" (gradient boosting on ~90 physics features) is
trained and tested on windows from our own generators, placed and impaired by
the shared impairment module. The CNN (seed 4) is scored on the same classes,
restricted to the group (argmax among the group's classes only), on windows
made the same way but NOT placed (centred at 0 Hz): its most favourable
condition, because the FSK/PSK/QAM training data was centred.

Features (all blind: nothing from the generator is used):
  lines   peak-to-average of |FFT(y^n)|, n = 1, 2, 4, 8: squaring BPSK gives a
          tone, the 4th power QPSK, the 8th 8PSK; noise and analog never do
  cyclo   symbol-rate line in the spectrum of |y|^2 (linear digital
          modulations with pulse shaping), its height and position
  ifreq   instantaneous frequency: spread, kurtosis, 16-bin histogram (FSK
          levels), smoothness (Gaussian shaping), its own spectrum (FM message)
  amp     envelope spread, kurtosis, 12-bin histogram, share near zero (OQPSK
          and MSK never pass through zero)
  delay   strongest autocorrelation peak at 16..320 samples and its delay
          (OFDM cyclic prefix, FMCW / chirp repetition)
  spec    occupied width, flatness, number of spectral peaks (multitone),
          symmetry about the centre (SSB)
  time    power in 16 blocks: max/min, on-fraction, single onset (reactive)

    python scripts/probe_specialists.py --n 200 --snr 15
Writes results/accuracy_plan/specialists_<snr>dB.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import expert_features as ef  # noqa: E402
from src.config import DATASET_CLASSES  # noqa: E402
from src.data import impairments as imp  # noqa: E402
from src.generators import analog, apsk, digital, jam, multicarrier, pulse, radar, spread  # noqa: E402

GEN = {c: m for m in (analog, apsk, digital, jam, multicarrier, pulse, radar, spread) for c in m.CLASSES}
GROUPS = {
    "PSK order": ["BPSK", "QPSK", "OQPSK", "8PSK", "16PSK", "32PSK"],
    "QAM sizes": ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM"],
    "APSK vs QAM vs band noise": ["16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "JAM_PARTIAL_BAND"],
    "FSK family": ["2FSK", "4FSK", "8FSK", "GFSK", "GMSK", "MSK", "CPFSK"],
    "Analog vs tones": ["AM_DSB_WC", "AM_DSB_SC", "AM_SSB_WC", "AM_SSB_SC", "WBFM", "NBFM", "PM", "JAM_TONE",
                        "JAM_SPOT", "AFSK", "OOK"],
    "OFDM vs noise jamming": ["OFDM", "LTE_LIKE", "NR_LIKE", "JAM_BARRAGE", "JAM_PARTIAL_BAND"],
    "Radar shapes": ["RADAR_PULSE", "LFM_RADAR", "RADAR_BARKER", "RADAR_POLY_BARKER", "RADAR_FRANK", "RADAR_COSTAS",
                     "RADAR_FMCW"],
    "Jammer kinds": ["JAM_BARRAGE", "JAM_PARTIAL_BAND", "JAM_SPOT", "JAM_MULTITONE", "JAM_PULSED", "JAM_REACTIVE",
                     "JAM_CHIRP", "JAM_SWEPT", "NOISE_FLOOR"],
    "Pulse comms vs radar": ["PPM", "SSR_IFF", "AFSK", "RADAR_PULSE", "RADAR_BARKER"],
}
FS = 3.2e6


def _pa_db(x):
    a = np.abs(x)
    return float(10 * np.log10((a.max() + 1e-12) / (a.mean() + 1e-12)))


def physics(z):
    z = np.asarray(z, complex)
    y, occ = ef.channelize(z)
    y = y / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
    n = len(y)
    f = []
    # lines: power-law spectral lines
    for p in (1, 2, 4, 8):
        S = np.abs(np.fft.fft(y ** p * np.hanning(n)))
        f += [_pa_db(S), float(np.argmax(S) / n)]
    # cyclo: symbol-rate line in |y|^2
    e = np.abs(y) ** 2
    E = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(n)))
    E[:3] = 0
    k = int(np.argmax(E))
    f += [_pa_db(E), k / n, float(E[k] / (np.median(E) + 1e-12))]
    # ifreq
    fi = np.diff(np.unwrap(np.angle(y))) / np.pi            # -1..1 = +-fs/2
    fi_c = fi - np.median(fi)
    sd = fi_c.std() + 1e-12
    f += [float(sd), float(np.mean(fi_c ** 4) / sd ** 4), float(np.mean(np.abs(np.diff(fi_c))) / sd)]
    h, _ = np.histogram(np.clip(fi_c / (3 * sd), -1, 1), bins=16, range=(-1, 1), density=True)
    f += list(h)
    FI = np.abs(np.fft.rfft(fi_c * np.hanning(len(fi_c))))
    FI = FI / (FI.sum() + 1e-12)
    f += [float(FI[1:8].sum()), float(FI[8:32].sum()), float(FI[32:96].sum()), float(FI[96:].sum())]
    # amp
    a = np.abs(y)
    f += [float(a.std()), float(np.mean((a - a.mean()) ** 4) / (a.var() ** 2 + 1e-12)), float(np.mean(a < 0.2))]
    ha, _ = np.histogram(a, bins=12, range=(0, 2.4), density=True)
    f += list(ha)
    # delay
    yc = y - y.mean()
    R = np.abs(np.fft.ifft(np.abs(np.fft.fft(yc, 2 * n)) ** 2))[:n]
    R = R / (R[0] + 1e-12)
    lag = 16 + int(np.argmax(R[16:321]))
    f += [float(R[lag]), lag / 320.0, float(np.sort(R[16:321])[-3:].mean())]
    # spec
    P = np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(len(z))))) ** 2
    Ps = np.convolve(P, np.ones(5) / 5, mode="same")
    floor = np.quantile(Ps, 0.3)
    band = Ps > 4 * floor
    inb = Ps[band] if band.any() else Ps
    flat = float(np.exp(np.mean(np.log(inb + 1e-20))) / (np.mean(inb) + 1e-20))
    peaks = int(np.sum((Ps[1:-1] > Ps[:-2]) & (Ps[1:-1] > Ps[2:]) & (Ps[1:-1] > 30 * floor)))
    Y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    m = len(Y) // 2
    sym = float(np.corrcoef(Y[m + 1:m + 64], Y[m - 1:m - 64:-1])[0, 1]) if Y.std() > 0 else 0.0
    f += [float(occ), flat, float(min(peaks, 20)), sym if np.isfinite(sym) else 0.0]
    # time
    bp = np.mean(np.abs(z.reshape(16, -1)) ** 2, axis=1)
    on = bp > 0.5 * bp.max()
    onset = int(np.argmax(on)) if on.any() else 0
    single = float(on[onset:].all() and onset > 0)
    f += [float(10 * np.log10(bp.max() / (bp.min() + 1e-12))), float(on.mean()), single,
          float(np.count_nonzero(on[1:] != on[:-1]))]
    # plus the 104 expert features the CNN already gets, for reference completeness
    return np.r_[f, ef.expert_features(z)].astype(np.float32)


def make(cls, snr, rng, place=True):
    skip = () if place else ("placement",)
    return imp.generate_impaired(GEN[cls], cls, snr, rng, skip=skip)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="windows per class")
    p.add_argument("--snr", type=float, default=15.0)
    p.add_argument("--classifier", default="results/classifier/physics_aug_best.pt")
    p.add_argument("--groups", nargs="*", default=None)
    a = p.parse_args()
    from src.pipeline.classify import Classifier
    clf = Classifier([a.classifier])
    rng = np.random.default_rng(11)
    out = {}
    print(f"{'group':28s} {'classes':>7s} {'chance':>7s} {'CNN in-group (centred)':>24s} {'specialist (placed)':>20s}")
    for name, classes in GROUPS.items():
        if a.groups and name not in a.groups:
            continue
        X, y, cnn_hits, per = [], [], [], {}
        idx = [DATASET_CLASSES.index(c) for c in classes]
        for k, c in enumerate(classes):
            for _ in range(a.n):
                X.append(physics(make(c, a.snr, rng, place=True)))
                y.append(k)
            wc = np.array([make(c, a.snr, rng, place=False) for _ in range(a.n)])
            pr = clf(wc)[:, idx]
            hits = pr.argmax(1) == k
            cnn_hits += list(hits)
            per[c] = {"cnn": float(hits.mean())}
        X, y = np.array(X), np.array(y)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=0, stratify=y)
        sp = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(X[tr], y[tr])
        pred = sp.predict(X[te])
        for k, c in enumerate(classes):
            m = y[te] == k
            per[c]["specialist"] = float((pred[m] == k).mean())
        r = {"classes": classes, "chance": 1 / len(classes), "cnn_in_group": float(np.mean(cnn_hits)),
             "specialist": float((pred == y[te]).mean()), "per_class": per}
        out[name] = r
        print(f"{name:28s} {len(classes):7d} {r['chance']:7.0%} {r['cnn_in_group']:24.0%} {r['specialist']:20.0%}",
              flush=True)
        weak = sorted(per.items(), key=lambda kv: kv[1]["specialist"])[:3]
        print("      weakest for the specialist: " + ", ".join(f"{c} {v['specialist']:.0%} (CNN {v['cnn']:.0%})"
                                                         for c, v in weak), flush=True)
    path = Path(f"results/accuracy_plan/specialists_{int(a.snr)}dB.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"snr_db": a.snr, "n_per_class": a.n, "groups": out}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
