"""End-to-end reference: a fixed IQ capture and the per-window probabilities
the REAL Python pipeline produces for it (sliding_windows -> preprocess_window
-> 5-model ensemble -> sigmoid -> average).

web/test/pipeline_check.mjs feeds the same capture through the JS pipeline
(preprocessWindow -> stftMag -> onnxruntime-web -> average) and must match.
This is the check that catches preprocessing/normalization mistakes, which a
STFT-only or ONNX-only test cannot see.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.config import CFG, CLASSES
from src.scenarios import CASES, build_scenario
from src.ui.app_models import EnsembleModel, _load_one, ensemble_paths
from src.timeline import classify_capture

HOP = 512

# Short capture -- 62 windows is plenty to catch a systematic error, and
# keeps the JSON small enough to hand to Node.
iq, segments = build_scenario(total_duration=0.01, snr_db=2, seed=7,
                               script=CASES["All three"])

model = EnsembleModel([_load_one(p) for p in ensemble_paths()]).eval()
result = classify_capture(iq, model, hop=HOP)

from src.measure import noise_floor_power, occupancy
from src.ui.session import CaptureSession
from src.config import resolve_multilabel_thresholds

session = CaptureSession(
    iq=np.asarray(iq), result=result, source="scenario", truth=segments,
    snr_known=True, true_snr_db=2, noise_power=noise_floor_power(iq),
    thresholds=dict(zip(CLASSES, resolve_multilabel_thresholds())),
)


def dump_events(events):
    return [{"startUs": e.start_us, "endUs": e.end_us,
             "durationUs": e.duration_us, "classes": list(e.classes),
             "peak": {k: float(v) for k, v in e.peak.items()}}
            for e in events]


out = {
    "iq_re": np.real(iq).tolist(),
    "iq_im": np.imag(iq).tolist(),
    "hop": HOP,
    "classes": CLASSES,
    "n_windows": int(result.n_windows),
    "probs": result.probs.tolist(),
    "starts": result.starts.tolist(),
    # display-layer results, both modes -- smoothing/gate/hold are the
    # intricate part and the part most likely to drift in a port
    "smoothed": {
        "events": dump_events(session.emitter_events(smoothed=True)),
        "tiers": session.tiers(smoothed=True),
    },
    "raw": {
        "events": dump_events(session.emitter_events(smoothed=False)),
        "tiers": session.tiers(smoothed=False),
    },
    # MEASURED
    "occupancy": occupancy(iq),
    "noise_floor_power": noise_floor_power(iq),
}
path = Path(__file__).parent / "pipeline_reference.json"
path.write_text(json.dumps(out))
print(f"wrote {path}  ({result.n_windows} windows, {len(iq)} samples)")
