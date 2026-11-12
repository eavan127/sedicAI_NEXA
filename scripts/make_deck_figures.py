"""
Figures for the November slides (presentation/FOI26/), made from the real
generators, the real model (results/classifier/physics_best.pt) and the real
Morse decoder, so every picture shows what NEXA actually does.

Writes PNGs to presentation/FOI26/assets/img/ and the numbers they show to
presentation/FOI26/assets/img/figures.json.

    python scripts/make_deck_figures.py
    python scripts/make_deck_figures.py --ckpt results/classifier/physics_aug_best.pt --out web/img/foi26
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_specialists import make  # noqa: E402
from src.config import CLASS_GROUPS, DATASET_CLASSES as C  # noqa: E402
from src.data.merge import normalize_batch  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

OUT = ROOT / "presentation" / "FOI26" / "assets" / "img"
CKPT = ROOT / "results" / "classifier" / "physics_best.pt"
INK, MUTE, BLUE, AMBER, GREEN, RED, TEAL, NAVY = ("#1A1A1A", "#5E5E5E", "#2E7DB9", "#C77A00", "#457900",
                                                   "#B0281A", "#1D7A68", "#22306B")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#BBBBBB",
                     "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
FS = 3.2e6
REP = {"JAMMING": "JAM_SWEPT", "RADAR": "LFM_RADAR", "PULSE_COMMS": "SSR_IFF", "SPREAD": "FHSS",
       "ANALOG": "WBFM", "ASK": "OOK", "MULTICARRIER": "LTE_LIKE", "APSK": "16APSK", "FSK": "4FSK",
       "PSK": "QPSK", "QAM": "64QAM"}
NICE = {"JAMMING": "Jammers", "RADAR": "Radar", "PULSE_COMMS": "Pulse comms", "SPREAD": "Spread spectrum",
        "ANALOG": "Analog voice", "ASK": "ASK", "MULTICARRIER": "OFDM / LTE / NR", "APSK": "APSK", "FSK": "FSK",
        "PSK": "PSK", "QAM": "QAM", "NOISE": "Noise floor"}


def spec(z, n_fft=64, hop=4):
    """|STFT| in dB, frequency centred, like the model's STFT branch (n_fft 64)."""
    w = np.hanning(n_fft)
    frames = [np.fft.fftshift(np.fft.fft(z[i:i + n_fft] * w)) for i in range(0, len(z) - n_fft + 1, hop)]
    m = 20 * np.log10(np.abs(np.array(frames)).T + 1e-9)
    return m - m.max()


def show_spec(ax, z, title=None, color=INK):
    ax.imshow(spec(z), aspect="auto", origin="lower", cmap="magma", vmin=-45, vmax=0,
              extent=[0, len(z) / FS * 1e6, -1.6, 1.6])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=4)


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = ck["meta"]
    net = AMC_CNN(num_classes=len(C), input_len=m["input_len"], stft_n_fft=m["stft_n_fft"], **m["model_flags"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def fig_gallery(rng):
    fams = list(REP)
    fig, axes = plt.subplots(2, 6, figsize=(15, 4.6))
    for ax, g in zip(axes.flat, fams + ["NOISE"]):
        cls = REP.get(g, "NOISE_FLOOR")
        show_spec(ax, make(cls, 20, rng, place=True), NICE[g])
        ax.text(0.5, -0.08, cls, transform=ax.transAxes, ha="center", va="top", fontsize=10, color=MUTE)
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    fig.savefig(OUT / "gallery_families.png", transparent=True)
    plt.close(fig)


def fig_snr(rng):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
    for ax, snr in zip(axes, (-10, 0, 10, 20)):
        show_spec(ax, make("LFM_RADAR", snr, rng, place=True), f"{snr:+d} dB")
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(OUT / "snr_ladder.png", transparent=True)
    plt.close(fig)


def fig_pipeline(net, rng):
    """One JAM_DRFM window (two emitters: radar + its jammer copy) through every step."""
    best = None
    for _ in range(60):
        z = make("JAM_DRFM", 18, rng, place=True)
        x = torch.from_numpy(normalize_batch(z[None]))
        with torch.no_grad():
            p = torch.sigmoid(net(x))[0].numpy()
        top = np.argsort(p)[::-1]
        if C[top[0]] in ("JAM_DRFM",) or C[top[1]] == "JAM_DRFM":
            score = p[top[1]]
            if best is None or score > best[0]:
                best = (score, z, p)
        if best is not None and best[0] > 0.5:
            break
    _, z, p = best
    x = torch.from_numpy(normalize_batch(z[None]))
    t = np.arange(len(z)) / FS * 1e6

    # 1 raw IQ and 2 normalised
    fig, ax = plt.subplots(2, 1, figsize=(6, 3.2), sharex=True)
    ax[0].plot(t, z.real, color=BLUE, lw=.8, label="I"); ax[0].plot(t, z.imag, color=AMBER, lw=.8, label="Q")
    ax[0].set_ylabel("raw"); ax[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    xn = x[0].numpy()
    ax[1].plot(t, xn[0], color=BLUE, lw=.8); ax[1].plot(t, xn[1], color=AMBER, lw=.8)
    ax[1].set_ylabel("normalised"); ax[1].set_xlabel("time (µs)")
    fig.savefig(OUT / "pipe_1_iq.png", transparent=True); plt.close(fig)

    # 3 STFT
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.imshow(spec(z), aspect="auto", origin="lower", cmap="magma", vmin=-45, vmax=0,
              extent=[0, 160, -1.6, 1.6])
    ax.set_xlabel("time (µs)"); ax.set_ylabel("MHz")
    fig.savefig(OUT / "pipe_2_stft.png", transparent=True); plt.close(fig)

    # 3b spectrum branch input: log power spectrum of the whole window
    P = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(len(z))))) ** 2 + 1e-12)
    P = np.convolve(np.pad(P, 3, mode="edge"), np.ones(7) / 7, mode="valid"); P -= P.max()
    f = np.linspace(-1.6, 1.6, len(z), endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.fill_between(f, P, -40, color=TEAL, alpha=.18, lw=0); ax.plot(f, P, color=TEAL, lw=1.4)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-35, 2)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("power (dB)")
    fig.savefig(OUT / "pipe_2b_spectrum.png", transparent=True); plt.close(fig)

    # 4 features + attention weights
    grab = {}
    net.attn_pool.register_forward_hook(lambda m, i, o: grab.update(x=i[0], power=i[1]))
    with torch.no_grad():
        net(x)
    fused, power = grab["x"], grab["power"]
    energy = F.adaptive_avg_pool1d(power, fused.shape[-1])
    with torch.no_grad():
        w = torch.softmax(net.attn_pool.score(torch.cat([fused, energy], 1)), dim=2)[0, 0].numpy()
    fmap = fused[0].numpy()
    fig, ax = plt.subplots(2, 1, figsize=(6, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax[0].imshow(fmap[np.argsort(fmap.std(1))[::-1][:48]], aspect="auto", cmap="viridis",
                 extent=[0, 160, 0, 48])
    ax[0].set_ylabel("features"); ax[0].set_yticks([])
    pw = power[0, 0].numpy()
    ax[1].fill_between(t, pw / pw.max(), color="#DDDDDD", label="signal power")
    ax[1].plot(t, w / w.max(), color=RED, lw=1.6, label="attention weight")
    ax[1].legend(loc="upper right", fontsize=9, frameon=False); ax[1].set_yticks([]); ax[1].set_xlabel("time (µs)")
    fig.savefig(OUT / "pipe_3_attention.png", transparent=True); plt.close(fig)

    # 5 physics numbers (normalised by the model's own BatchNorm)
    with torch.no_grad():
        ph = net.physics_norm(net.physics_branch(x))[0].numpy()
    fig, ax = plt.subplots(figsize=(6, 1.9))
    cols = [BLUE] * 4 + [TEAL] * 3 + [AMBER] * 23 + [GREEN] * 15 + [NAVY] * 3 + [RED] * 8
    ax.bar(range(len(ph)), ph, color=cols[:len(ph)], width=.8)
    ax.axhline(0, color="#999999", lw=.6); ax.set_xticks([]); ax.set_ylabel("z-score")
    fig.savefig(OUT / "pipe_4_physics.png", transparent=True); plt.close(fig)

    # 6 sigmoid outputs, multi-label
    top = np.argsort(p)[::-1][:8]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    colors = [GREEN if p[k] >= 0.5 else "#BBBBBB" for k in top]
    ax.barh(range(8)[::-1], p[top], color=colors)
    ax.set_yticks(range(8)[::-1]); ax.set_yticklabels([C[k] for k in top], fontsize=10)
    ax.axvline(0.5, color=RED, ls="--", lw=1.2); ax.text(0.505, 7.4, "0.5", color=RED, fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("probability (independent sigmoid per class)")
    fig.savefig(OUT / "pipe_5_multilabel.png", transparent=True); plt.close(fig)
    return {"example": "JAM_DRFM at 18 dB", "top": [[C[k], round(float(p[k]), 3)] for k in top]}


def fig_vote():
    n = [1, 2, 4, 8, 16, 32, 64]
    # physics_aug_best (current classifier) on f993d8ebc417, calibrate_week1 vote proxy, 2026-11-10
    all_snr = [47.3, 58.4, 66.5, 70.7, 73.0, 74.2, 74.8]
    high = [61.7, 69.9, 76.8, 77.5, 79.4, 79.9, 79.1]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.plot(n, high, "o-", color=GREEN, lw=2.2, label="≥ 10 dB")
    ax.plot(n, all_snr, "o-", color=BLUE, lw=2.2, label="all SNR")
    ax.axhline(80, color=RED, ls="--", lw=1); ax.text(1.05, 81.2, "80%", color=RED, fontsize=9)
    ax.set_xscale("log", base=2); ax.set_xticks(n); ax.set_xticklabels(n)
    ax.set_xlabel("windows voted (× 160 µs)"); ax.set_ylabel("exact class (%)"); ax.set_ylim(40, 90)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "vote_curve.png", transparent=True); plt.close(fig)


def fig_scene(rng):
    from src.detect.scenes import make_scene
    for _ in range(40):
        iq, img, boxes, meta = make_scene(rng, n_emitters=5, snr_db=None)
        if len(boxes) >= 4:
            break
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(img, aspect="auto", cmap="magma", origin="upper")
    for (x0, y0, x1, y1), cls in zip(boxes, meta["cls"]):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#7CFC00", lw=1.6))
        ax.text(x0, y0 - 2, cls, color="#7CFC00", fontsize=8, fontweight="bold", va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "scene_boxes.png", transparent=True); plt.close(fig)
    return iq, img, boxes


def _thumb(name, draw, size=(3.2, 1.8)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(OUT / f"t_{name}.png", transparent=True, dpi=130)
    plt.close(fig)


def fig_thumbs(net, rng, scene, pipe_top):
    """Small pictures for the boxes of the architecture slides (04, 05)."""
    iq, img, boxes = scene
    _thumb("capture", lambda ax: (ax.plot(iq[:6000].real, color=BLUE, lw=.4), ax.plot(iq[:6000].imag + 0, color=AMBER, lw=.4, alpha=.7)))
    _thumb("spectrogram", lambda ax: ax.imshow(img, aspect="auto", cmap="magma"))

    def det(ax):
        ax.imshow(img, aspect="auto", cmap="magma")
        for x0, y0, x1, y1 in boxes:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#7CFC00", lw=1.4))
    _thumb("detector", det)
    _thumb("cut", lambda ax: ax.imshow(spec(make("LFM_RADAR", 18, rng, place=False)), aspect="auto", origin="lower",
                                       cmap="magma", vmin=-45, vmax=0))

    def cls(ax):
        names, vals = zip(*pipe_top[:4])
        ax.barh(range(4)[::-1], vals, color=[GREEN if v >= .5 else "#BBBBBB" for v in vals])
        for k, (n, v) in enumerate(zip(names, vals)):
            ax.text(0.01, 3 - k, n, va="center", fontsize=11, fontweight="bold", color=INK)
        ax.axvline(.5, color=RED, ls="--", lw=1.2); ax.set_xlim(0, 1)
    _thumb("classifier", cls)

    k = C.index("QPSK")
    W = np.array([make("QPSK", 6, rng, place=True) for _ in range(16)])
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(normalize_batch(W)))).numpy()
    P = np.clip(P, 1e-4, 1 - 1e-4)
    L = np.log(P / (1 - P))
    vote = [1 / (1 + np.exp(-L[:i + 1].mean(0)))[k] for i in range(16)]
    right = [L[:i + 1].mean(0).argmax() == k for i in range(16)]

    def trk(ax):
        ax.bar(range(16), P[:, k], color=[GREEN if P[i].argmax() == k else "#CCCCCC" for i in range(16)], width=.7)
        ax.plot(range(16), vote, color=NAVY, lw=2.4)
        ax.set_ylim(0, max(1e-3, P[:, k].max()) * 1.15)
    _thumb("track", trk)

    from src.pipeline import analog, analog_synth, demod
    y, _ = analog_synth.make("AM", seconds=1.5, snr_db=25, rng=np.random.default_rng(0))
    kind, feats, _ = analog.classify(y, analog.FA)
    f = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / analog.FA))
    S = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(y * np.hanning(len(y))))) + 1e-9)
    m = np.abs(f) < 6000
    _thumb("zoom", lambda ax: ax.plot(f[m] / 1e3, S[m] - S[m].max(), color=TEAL, lw=.6))
    a = demod.demodulate(y, analog.FA, kind, feats)
    _thumb("demod", lambda ax: ax.plot(np.arange(len(a)) / 16000, a, color=TEAL, lw=.3))
    return {"zoom_kind": kind, "track_vote_right_after": int(np.argmax(right)) + 1 if any(right) else None,
            "track_window_right": int(sum(P[i].argmax() == k for i in range(16)))}


def fig_morse():
    from src.intel import morse
    fs = 8000
    x = morse.synth("CQ CQ DE NEXA K", wpm=18, tone=700, fs=fs, snr_db=12, rng=np.random.default_rng(3))
    tone, _ = morse.tone_and_purity(x, fs)
    env = morse.envelope(x, fs, tone)
    keyed, snr = morse.key_track(env)
    segs = morse.decode_keyed(keyed)
    text = " ".join(s["text"] for s in segs)
    tt = np.arange(len(env)) / morse.ENV_RATE
    fig, ax = plt.subplots(2, 1, figsize=(9, 2.6), sharex=True, gridspec_kw={"height_ratios": [1, .6]})
    ax[0].plot(np.arange(len(x)) / fs, x, color="#BBBBBB", lw=.3)
    ax[0].plot(tt, env / env.max() * np.abs(x).max(), color=TEAL, lw=1.2)
    ax[0].set_yticks([]); ax[0].set_ylabel("audio")
    ax[1].fill_between(tt, keyed.astype(float), step="mid", color=NAVY)
    ax[1].set_yticks([]); ax[1].set_ylabel("key"); ax[1].set_xlabel("seconds")
    fig.savefig(OUT / "morse_decode.png", transparent=True); plt.close(fig)
    _thumb("morse", lambda ax: ax.fill_between(tt[:int(4.5 * morse.ENV_RATE)], keyed[:int(4.5 * morse.ENV_RATE)].astype(float),
                                               step="mid", color=NAVY), size=(3.2, 1.0))
    return {"text": text, "wpm": segs[0]["wpm"] if segs else None, "tone_hz": round(tone)}


def main():
    global OUT, CKPT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(CKPT), help="classifier checkpoint")
    p.add_argument("--out", default=str(OUT), help="folder for the PNGs and figures.json")
    a = p.parse_args()
    CKPT, OUT = Path(a.ckpt), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    net = load_model()
    out = {}
    fig_gallery(rng)
    fig_snr(rng)
    out["pipeline"] = fig_pipeline(net, rng)
    fig_vote()
    scene = fig_scene(rng)
    out["morse"] = fig_morse()
    out["thumbs"] = fig_thumbs(net, rng, scene, out["pipeline"]["top"])
    (OUT / "figures.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
