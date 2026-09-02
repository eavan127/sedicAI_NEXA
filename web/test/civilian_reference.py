"""Reference for the civilian scenario path: the exported library must match
what civilian_library() returns, and load_scenario's SNR-capping arithmetic
must match what the JS buildScenario computes.

The waveform itself cannot be compared sample-for-sample (the two RNGs draw
different window sequences by design -- see generators.js's module comment),
so this pins the parts that are deterministic: the library contents, and the
achieved/capped SNR for every civilian case at every SNR bin.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.config import CFG
from src.scenarios import CASES, CIVILIAN
from src.ui.session import CLEANEST_LIBRARY_SNR_DB, civilian_library

lib = civilian_library()

cases = {}
for name, script in CASES.items():
    needs = any(c in CIVILIAN for c, _, _ in script)
    per_snr = {}
    for snr_db in CFG["snr_bins_db"]:
        library_snr = CLEANEST_LIBRARY_SNR_DB if needs else None
        true_snr = min(snr_db, library_snr) if needs else snr_db
        per_snr[str(snr_db)] = {
            "true_snr_db": float(true_snr),
            "snr_capped": bool(needs and snr_db > library_snr),
        }
    cases[name] = {"needs_library": needs, "per_snr": per_snr}

out = {
    "library_snr_db": float(CLEANEST_LIBRARY_SNR_DB),
    "library": {cls: {"n": int(w.shape[0]),
                       # checksum the exported bytes agree window-for-window
                       "sum": float(np.sum(w)), "absmax": float(np.max(np.abs(w)))}
                 for cls, w in lib.items()},
    "cases": cases,
    "case_names": list(CASES.keys()),
}
path = Path(__file__).parent / "civilian_reference.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {path}  ({len(CASES)} cases, "
      f"{ {c: v['n'] for c, v in out['library'].items()} })")
