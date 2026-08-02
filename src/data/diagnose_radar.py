"""
Which radar examples does the model get wrong?

The confusion matrix shows radar being predicted as JAMMING, and barrage jamming
being predicted as RADAR — a bidirectional confusion. The suspicion is
bandwidth: our chirps reach 1.5 MHz against a 1.6 MHz Nyquist, so a wide chirp
smears energy across nearly the whole band, which is what barrage noise looks
like.

This sweeps radar accuracy against bandwidth and duty cycle to find out.
Non-invasive: generates fresh examples, probes the existing checkpoint.

Usage:
    python -m src.data.diagnose_radar
"""
import numpy as np
import torch

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT
from src.data.preprocess import add_awgn, preprocess_window
from src.generators.radar import embed_pulse_train, generate_lfm_chirp_iq
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = CFG["signal"]["fs"]
WINDOW = CFG["signal"]["window_len"]
TOTAL = CFG["signal"]["total_duration"]
RADAR_IDX = CLASS_TO_IDX["LFM_RADAR"]


def _radar_with(bandwidth, duty, rng):
    """One radar example at a CHOSEN bandwidth and duty cycle, so we can sweep
    them independently instead of taking whatever the sampler happens to draw."""
    cfg = CFG["radar"]
    pulse_width = rng.uniform(*cfg["pulse_width_s"])
    pri = pulse_width / duty
    time_delay = rng.uniform(*cfg["time_delay_s"])

    f_start = -bandwidth / 2 if rng.random() > 0.5 else bandwidth / 2
    bw = bandwidth if f_start < 0 else -bandwidth

    n_pulses = (int(rng.integers(*cfg["n_pulses"]))
                if rng.random() < cfg["burst_fraction"] else None)
    pulse = generate_lfm_chirp_iq(FS, pulse_width, bw, f_start)
    return embed_pulse_train(pulse, pri, FS, TOTAL, time_delay, n_pulses)


def _load_model():
    path = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path} — train first.")
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def _accuracy(model, gen, n, rng):
    batch = []
    for _ in range(n):
        snr = float(rng.choice(CFG["snr_bins_db"]))
        batch.append(preprocess_window(add_awgn(gen()[:WINDOW], snr, rng=rng)))
    with torch.no_grad():
        preds = model(torch.tensor(np.stack(batch)).to(DEVICE)).argmax(1).cpu().numpy()
    acc = float((preds == RADAR_IDX).mean())

    wrong = preds[preds != RADAR_IDX]
    worst = "-"
    if wrong.size:
        idx, cnt = np.unique(wrong, return_counts=True)
        worst = f"{CLASSES[idx[cnt.argmax()]]} ({cnt.max()}/{n})"
    return acc, worst


def sweep(n_per=150, seed=0):
    model = _load_model()
    rng = np.random.default_rng(seed)
    cfg = CFG["radar"]

    print("Radar accuracy vs BANDWIDTH  (duty fixed at 10%)")
    print(f"{'bandwidth':<16}{'correct':>9}   most common mistake")
    print("-" * 58)
    lo, hi = cfg["bandwidth_hz"]
    for bw in np.linspace(lo, hi, 6):
        acc, worst = _accuracy(model, lambda b=bw: _radar_with(b, 0.10, rng), n_per, rng)
        flag = "  <-- WEAK" if acc < 0.85 else ""
        print(f"{bw/1e3:>10.0f} kHz {acc:>8.1%}   {worst}{flag}")

    print()
    print("Radar accuracy vs DUTY CYCLE  (bandwidth fixed at 400 kHz)")
    print(f"{'duty cycle':<16}{'correct':>9}   most common mistake")
    print("-" * 58)
    for duty in (0.02, 0.05, 0.10, 0.25, 0.50, 0.90):
        acc, worst = _accuracy(model, lambda d=duty: _radar_with(400e3, d, rng), n_per, rng)
        flag = "  <-- WEAK" if acc < 0.85 else ""
        print(f"{duty:>10.0%}     {acc:>8.1%}   {worst}{flag}")

    print()
    print("If accuracy falls as bandwidth rises, wide chirps are being read as")
    print("barrage noise — narrow bandwidth_hz in configs/default.yaml.")
    print("If it falls as duty rises, near-continuous radar looks like a jammer")
    print("— lower max_duty_cycle.")


def real_vs_synthetic(n=400, seed=1):
    """Does the model handle REAL RadChar radar as well as our synthetic kind?

    This is the open question left by capping duty at 15%. RadChar runs at
    44-94% duty — exactly the band we stopped generating — so half the radar
    training data still lives in the region a synthetic-only sweep says is
    unlearnable. If real high-duty examples score well while synthetic ones do
    not, the cap was right and RadChar carries that regime. If both fail, the
    class boundary is broken there regardless of source.
    """
    from src.data.radchar import load_radchar_lfm

    model = _load_model()
    rng = np.random.default_rng(seed)

    print("Radar accuracy: REAL (RadChar) vs SYNTHETIC (ours)")
    print(f"{'source':<28}{'correct':>9}   most common mistake")
    print("-" * 62)

    # --- real: already carries its own noise at its labelled SNR
    try:
        real = load_radchar_lfm(per_snr=n // len(CFG["snr_bins_db"]),
                                 snr_bins=CFG["snr_bins_db"])
    except FileNotFoundError:
        print("  RadChar not found — skipping the real half.")
        real = []

    if real:
        batch = np.stack([preprocess_window(iq) for iq, _, _ in real])
        with torch.no_grad():
            preds = model(torch.tensor(batch).to(DEVICE)).argmax(1).cpu().numpy()
        acc = float((preds == RADAR_IDX).mean())
        wrong = preds[preds != RADAR_IDX]
        worst = "-"
        if wrong.size:
            idx, cnt = np.unique(wrong, return_counts=True)
            worst = f"{CLASSES[idx[cnt.argmax()]]} ({cnt.max()}/{len(real)})"
        print(f"{'RadChar (real, 44-94% duty)':<28}{acc:>8.1%}   {worst}")

    # --- synthetic: our generator, whatever the config currently allows
    from src.generators.radar import random_radar_example
    acc, worst = _accuracy(model, lambda: random_radar_example(rng=rng), n, rng)
    duty_hi = CFG["radar"]["max_duty_cycle"]
    print(f"{f'ours (synthetic, <={duty_hi:.0%} duty)':<28}{acc:>8.1%}   {worst}")

    print()
    print("If real scores well and synthetic does not, RadChar is carrying the")
    print("high-duty regime and the cap was the right call. If real also fails,")
    print("high-duty radar is ambiguous with FHSS no matter where it comes from.")


if __name__ == "__main__":
    import sys

    if "--sources" in sys.argv:
        real_vs_synthetic()
    else:
        sweep()
        print()
        real_vs_synthetic()
