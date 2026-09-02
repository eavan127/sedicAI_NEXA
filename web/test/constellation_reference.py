"""Reference for the constellation panel.

Splits deliberately into two parts, because they can be held to different
bars:

  deterministic -- rrc_taps, carrier_offset, recover_symbols, _normalized_c42,
                   constellation_order, civilian_windows. These must match to
                   float precision.
  stochastic    -- cluster_score, which draws from numpy's PCG64 (k-means++
                   seeding and 15 null resamples). JS cannot reproduce that
                   bit stream, so the check is statistical: same band, and
                   close values, across many real windows.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.measure import constellation_order, noise_floor_power
from src.scenarios import CASES
from src.ui.plots import (CONSTELLATION_ORDER, cluster_score,
                           cluster_score_band, carrier_offset,
                           _symbols_per_window, recover_symbols, rrc_taps)
from src.ui.session import (CLEANEST_LIBRARY_SNR_DB, civilian_library,
                             load_scenario)
from src.ui.app_models import EnsembleModel, _load_one, ensemble_paths

WINDOW_LEN = CFG["signal"]["window_len"]

# A civilian-bearing capture, so the panel has real windows to work on.
model = EnsembleModel([_load_one(p) for p in ensemble_paths()]).eval()
session = load_scenario(model, total_duration=0.02, hop=512, snr_db=10,
                         seed=11, case="Civilian only")

taps = rrc_taps(8)

windows, per_window = [], []
for i, start in enumerate(session.result.starts):
    w = session.iq[int(start):int(start) + WINDOW_LEN]
    if len(w) < WINDOW_LEN:
        continue
    points, offset, phase = recover_symbols(w)
    recovered = len(points) < len(w)
    entry = {
        "index": int(i),
        "offset": float(offset),
        "phase": int(phase),
        "n_points": int(len(points)),
        "recovered": bool(recovered),
        "points_re": np.real(points).tolist(),
        "points_im": np.imag(points).tolist(),
        "carrier_offset_raw": float(carrier_offset(w)),
    }
    # cluster_score at every order this project actually asks for
    entry["cluster_score"] = {
        str(order): float(cluster_score(points, order))
        for order in sorted(set(CONSTELLATION_ORDER.values()))
        if len(points) >= order
    }
    entry["cluster_band"] = {
        k: cluster_score_band(v) for k, v in entry["cluster_score"].items()
    }
    windows.append({"re": np.real(w).tolist(), "im": np.imag(w).tolist()})
    per_window.append(entry)

picks = session.civilian_windows(count=4, smoothed=True)
all_qualifying = session.civilian_windows(count=len(session.result.probs),
                                           smoothed=True)
pooled = [session.iq[int(session.result.starts[i]):
                      int(session.result.starts[i]) + WINDOW_LEN]
           for i, _, _ in all_qualifying]
est = constellation_order(pooled, session.noise_power) if pooled else None

out = {
    "window_len": WINDOW_LEN,
    "rrc_taps": taps.tolist(),
    "symbols_per_window": int(_symbols_per_window(WINDOW_LEN)),
    "noise_power": float(session.noise_power),
    "thresholds": dict(zip(CLASSES, [float(v) for v in resolve_multilabel_thresholds()])),
    "n_windows": int(session.result.n_windows),
    "n_classes": len(CLASSES),
    # civilian_windows(smoothed=True) selects on the RESOLVED (smoothed)
    # probabilities, not the raw ones -- see CaptureSession._resolved. The
    # JS caller has to pass the same thing.
    "probs": session._resolved(True).probs.tolist(),
    "raw_probs": session.result.probs.tolist(),
    "windows": windows,
    "per_window": per_window,
    "civilian_picks": [{"index": int(i), "cls": c, "prob": float(p)} for i, c, p in picks],
    "n_qualifying": len(all_qualifying),
    "constellation_order": None if est is None else {
        "decision": est.decision, "mean_c42": float(est.mean_c42),
        "n_windows": int(est.n_windows), "margin": float(est.margin),
        "accuracy": est.accuracy,
        "snr_db": None if est.snr_db is None else float(est.snr_db),
    },
}
path = Path(__file__).parent / "constellation_reference.json"
path.write_text(json.dumps(out))
print(f"wrote {path}  ({len(per_window)} windows, "
      f"{len(picks)} picks, order={out['constellation_order']})")
