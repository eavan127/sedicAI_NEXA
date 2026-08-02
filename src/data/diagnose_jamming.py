"""
Which jamming sub-type is the model getting wrong?

The confusion matrix says JAMMING is weak (59 examples predicted as FHSS) but
not WHY, because random_jamming_example() picks barrage/tone/sweep and discards
which it chose. This probes the trained model with each sub-type separately, so
P4 knows what to fix instead of guessing.

Non-invasive: it does not touch the dataset or retrain. It generates fresh
examples and asks the existing checkpoint what it thinks.

Usage:
    python -m src.data.diagnose_jamming
"""
import numpy as np
import torch

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT
from src.data.preprocess import add_awgn, preprocess_window
from src.generators.fhss import random_fhss_example
from src.generators.jamming import (generate_barrage_jamming,
                                     generate_sweep_jamming,
                                     generate_tone_jamming)
from src.generators.radar import random_radar_example
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = CFG["signal"]["fs"]
WINDOW = CFG["signal"]["window_len"]
TOTAL = CFG["signal"]["total_duration"]
N = int(FS * TOTAL)


def _subtype_generators(rng):
    """One callable per jamming sub-type, plus the two classes it gets confused
    with, so the same probe reports on all of them."""
    return {
        "JAM barrage": lambda: generate_barrage_jamming(N, rng=rng),
        "JAM tone": lambda: generate_tone_jamming(
            FS, N, rng.uniform(-FS / 4, FS / 4, rng.integers(1, CFG["jamming"]["max_tones"] + 1))),
        "JAM sweep": lambda: generate_sweep_jamming(
            FS, TOTAL, rng.uniform(*CFG["jamming"]["sweep_bandwidth_hz"])),
        "FHSS (reference)": lambda: random_fhss_example(rng=rng),
        "RADAR (reference)": lambda: random_radar_example(rng=rng),
    }


TRUTH = {
    "JAM barrage": "JAMMING", "JAM tone": "JAMMING", "JAM sweep": "JAMMING",
    "FHSS (reference)": "FHSS", "RADAR (reference)": "LFM_RADAR",
}


def diagnose(n_per=200, model_path=None, seed=0):
    model_path = model_path or REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {model_path}\n"
            "Train first, or download best_model.pt from Colab into results/."
        )

    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    rng = np.random.default_rng(seed)
    gens = _subtype_generators(rng)

    print(f"Probing the trained model with {n_per} fresh examples per sub-type")
    print(f"across SNR bins {CFG['snr_bins_db']}\n")
    print(f"{'sub-type':<20}{'correct':>9}   most common mistake")
    print("-" * 62)

    for name, gen_fn in gens.items():
        batch = []
        for _ in range(n_per):
            snr = float(rng.choice(CFG["snr_bins_db"]))
            batch.append(preprocess_window(add_awgn(gen_fn()[:WINDOW], snr, rng=rng)))

        with torch.no_grad():
            preds = model(torch.tensor(np.stack(batch)).to(DEVICE)).argmax(1).cpu().numpy()

        want = CLASS_TO_IDX[TRUTH[name]]
        acc = float((preds == want).mean())

        wrong = preds[preds != want]
        if wrong.size:
            idx, cnt = np.unique(wrong, return_counts=True)
            worst = CLASSES[idx[cnt.argmax()]]
            detail = f"{worst}  ({cnt.max()}/{n_per})"
        else:
            detail = "-"

        flag = "  <-- WEAK" if acc < 0.9 and name.startswith("JAM") else ""
        print(f"{name:<20}{acc:>8.1%}   {detail}{flag}")

    print()
    print("Any jamming sub-type below 90% is what is dragging the class down.")
    print("Hand the failing sub-type to P4 — that is the generator to fix.")


if __name__ == "__main__":
    diagnose()
