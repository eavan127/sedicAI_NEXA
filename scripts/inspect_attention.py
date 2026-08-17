"""
Does the attention-pooling layer actually attend to the signal, or did it
learn nothing and end up roughly flat (= expensive average pooling)?

Generates a few real radar and FHSS examples (pulse timing is randomised per
example -- see config: time_delay_s -- so this also checks whether attention
tracks a MOVING pulse position, not just a fixed spot it memorised), captures
the attention weights via a forward hook, and plots weight-over-time next to
signal-amplitude-over-time so you can see with your own eyes whether they
line up.

Non-invasive: reads an existing checkpoint, generates fresh examples, does
not touch the dataset or retrain.

Usage:
    python scripts/inspect_attention.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, REPO_ROOT  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.radar import random_radar_example  # noqa: E402
from src.generators.fhss import random_fhss_example  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = CFG["signal"]["window_len"]


def main(model_path=None, snr_db=-6, n_examples=3, seed=0):
    model_path = model_path or REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No checkpoint at {model_path} -- train on this branch first "
            "(python -m src.train), or download one from Colab into results/."
        )

    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    if not hasattr(model, "attn_pool"):
        raise AttributeError(
            "This checkpoint's model has no attn_pool -- it's the old "
            "flatten()+Linear architecture, not the attention-pooling one."
        )

    # Grab the raw per-timestep scores as they're produced, before this
    # script applies the same softmax the model applies internally.
    captured = {}

    def hook(_module, _input, output):
        captured["raw_scores"] = output.detach().cpu()

    model.attn_pool.score.register_forward_hook(hook)

    rng = np.random.default_rng(seed)
    sources = {"LFM_RADAR": random_radar_example, "FHSS": random_fhss_example}

    fig, axes = plt.subplots(len(sources), n_examples, figsize=(4 * n_examples, 6), squeeze=False)

    for row, (cls_name, gen_fn) in enumerate(sources.items()):
        for col in range(n_examples):
            raw = gen_fn(rng=rng)[:WINDOW]
            noisy = add_awgn(raw, snr_db, rng=rng)
            arr = preprocess_window(noisy, WINDOW)

            with torch.no_grad():
                model(torch.tensor(arr[None]).to(DEVICE))

            scores = captured["raw_scores"][0, 0]                  # (time,)
            weights = torch.softmax(scores, dim=0).numpy()
            amplitude = np.abs(raw[:WINDOW] if len(raw) >= WINDOW
                                else np.pad(raw, (0, WINDOW - len(raw))))
            amplitude = amplitude / (amplitude.max() + 1e-8)

            ax = axes[row][col]
            t = np.arange(WINDOW)
            ax.plot(t, amplitude, color="gray", alpha=0.6, label="signal amplitude (norm.)")
            ax.plot(t, weights / (weights.max() + 1e-8), color="tab:red",
                    label="attention weight (norm.)")
            ax.set_title(f"{cls_name} #{col+1} @ {snr_db} dB", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    out = REPO_ROOT / CFG["paths"]["evals"] / "attention_inspection.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    print("\nRead it as: does the red attention curve rise where the grey")
    print("amplitude curve (the actual pulse/hop) rises, and stay low during")
    print("the quiet/noise stretches? If the red curve is roughly flat")
    print("everywhere, attention did not learn anything -- it's behaving")
    print("like plain average pooling despite the extra parameters.")


if __name__ == "__main__":
    main()
