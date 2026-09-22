"""
Report: how many RadChar LFM waveforms are actually usable per our SNR bin,
once bucketed by MEASURED (not labelled) SNR -- see radchar.py's
"measured (not labelled) SNR" section and diagnose_radchar_snr.py for why the
label can't be trusted directly.

This is a reporting/sanity-check CLI over the same measurement
load_radchar_lfm_by_measured_snr uses internally (and shares its cache at
radchar.RADCHAR_MEASURED_CACHE) -- it doesn't do anything build_dataset.py
doesn't already do when radchar_fraction > 0, it just prints the counts and
the mixture-eligibility ceiling so a human can sanity-check the composition
before committing to a full rebuild.

Usage:
    python -m src.data.rebucket_radchar
"""
import numpy as np

from src.config import CFG
from src.data.radchar import measure_all_lfm, RADCHAR_MEASURED_CACHE

OUR_BINS = np.array(CFG["snr_bins_db"], dtype=float)


def main(cap_per_label=None, tolerance_db=3.0):
    rows, measured, label = measure_all_lfm(cap_per_label=cap_per_label)
    print(f"{len(rows):,} LFM waveforms measured (cache: {RADCHAR_MEASURED_CACHE})\n")

    dist = np.abs(measured[:, None] - OUR_BINS[None, :])
    nearest_idx = dist.argmin(axis=1)
    keep = dist.min(axis=1) <= tolerance_db

    print(f"Composition after re-bucketing by MEASURED SNR, tolerance +/-{tolerance_db}dB:")
    print(f"{'our bin':>8} | {'usable RadChar waveforms':>25} | {'measured range actually used':>29}")
    print("-" * 70)
    for i, b in enumerate(OUR_BINS):
        sel = keep & (nearest_idx == i)
        n = sel.sum()
        if n == 0:
            print(f"{b:>8.0f} | {0:>25} | (none within tolerance)")
            continue
        lo, hi = measured[sel].min(), measured[sel].max()
        print(f"{b:>8.0f} | {n:>25,} | {lo:>6.2f} to {hi:>6.2f} dB")

    print(f"\nCeiling check: highest MEASURED SNR found = {measured.max():.2f} dB")
    for thr in (2, 6, 10):
        print(f"waveforms measuring above +{thr}dB: {(measured > thr).sum():,}")


if __name__ == "__main__":
    main()
