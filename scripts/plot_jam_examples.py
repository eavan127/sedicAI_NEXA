"""
One spectrogram per JAM class (src/generators/jam.py), clean and at an SNR,
saved as a single PNG -- a quick eye check that the classes look different.

    python scripts/plot_jam_examples.py
    python scripts/plot_jam_examples.py --snr 0 --seed 3 --out data/interim/jam_examples.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import impairments  # noqa: E402
from src.generators import jam  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snr", type=float, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/interim/jam_examples.png")
    args = p.parse_args()

    fig, axes = plt.subplots(2, len(jam.CLASSES), figsize=(2.2 * len(jam.CLASSES), 5))
    for col, cls in enumerate(jam.CLASSES):
        clean, _ = jam.generate_window(cls, None, np.random.default_rng(args.seed))
        noisy, _ = impairments.generate_impaired(jam, cls, args.snr, np.random.default_rng(args.seed))
        for row, (x, label) in enumerate(((clean, "clean"), (noisy, f"{args.snr:g} dB + impair"))):
            ax = axes[row, col]
            ax.specgram(x, NFFT=64, noverlap=48, Fs=jam.FS / 1e6, cmap="viridis")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(cls.replace("JAM_", ""), fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
