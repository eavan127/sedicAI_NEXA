"""End-to-end reference: a fixed IQ capture and the per-window probabilities
the REAL Python pipeline produces for it with the 60-class model the
dashboard runs (results/classifier/physics_best.pt, the checkpoint
src/pipeline/classify.py loads): sliding_windows -> preprocess_window ->
AMC_CNN -> sigmoid.

web/test/pipeline_check.mjs feeds the same capture through the JS pipeline
(preprocessWindow -> stftMag -> web/models/best_model.onnx) and must match.
This is the check that catches preprocessing/normalization mistakes, which a
STFT-only or ONNX-only test cannot see. web/test/analysis_check.mjs checks the
display layer (smoothing, gate, hold, events, tiers) against the Python
CaptureSession on the same probabilities.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import CFG
from src.pipeline.classify import Classifier
from src.scenarios import CASES, build_scenario
from src.timeline import classify_capture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_model import CKPT, CLASSES, THRESHOLDS, load_model, use_page_classes  # noqa: E402

HOP = 512
use_page_classes()

# Short capture -- 62 windows is plenty to catch a systematic error, and
# keeps the JSON small enough to hand to Node.
iq, segments = build_scenario(total_duration=0.01, snr_db=2, seed=7,
                               script=CASES["All three"])

model = load_model()
result = classify_capture(iq, model, hop=HOP)

# Same numbers the Signal Pipeline's Classifier gives for these windows.
iq_arr = np.asarray(iq)
cuts = np.stack([iq_arr[s:s + 512] for s in result.starts])
assert np.allclose(Classifier([CKPT], device="cpu")(cuts), result.probs, atol=1e-5)

from src.measure import noise_floor_power, occupancy
from src.ui.session import CaptureSession

session = CaptureSession(
    iq=np.asarray(iq), result=result, source="scenario", truth=segments,
    snr_known=True, true_snr_db=2, noise_power=noise_floor_power(iq),
    thresholds=THRESHOLDS,
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
