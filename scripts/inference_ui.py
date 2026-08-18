"""
Local inference UI: upload a raw IQ file (or pick a real held-out test
example), see its spectrogram, and get a real classification from the
trained model -- not a simulation.

This must run locally, not as a claude.ai Artifact: an Artifact is sandboxed
JS with no Python and no way to load a .pt checkpoint or call your real
model. Gradio gives the same "upload data, see the result" experience while
actually running your model.

Usage:
    python scripts/inference_ui.py
Then open the printed http://127.0.0.1:7860 link.
"""
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, REPO_ROOT  # noqa: E402
from src.data.preprocess import preprocess_window  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402
from src.evaluate import TIERS  # noqa: E402
from src.evaluate import evaluate as run_full_evaluation  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]
CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
EVALS_DIR = REPO_ROOT / CFG["paths"]["evals"]

# Shared light palette -- matches the Overwatch waterfall Artifact's tier
# colors, just on a light ground instead of dark, so the two tools still read
# as one product.
_BG = "#f6f8fa"
_PANEL = "#ffffff"
_GRID = "#e1e7ee"
_TEXT = "#17212b"
_TEXT_DIM = "#5b6b7c"

# Deepened versions of the artifact's tier hues -- the dark-ground colors
# (#4fd1c5 etc.) don't have enough contrast against a light background, so
# each is shifted darker while keeping the same hue, preserving the
# civilian/military/hostile/empty color language.
_TIER_COLOR = {"Civilian": "#0d9488", "Military": "#b45309",
               "Hostile": "#dc2626", "Empty": "#5b6b7c"}


def _style_light_axes(fig, axes):
    """Applies the console palette to a matplotlib figure this script owns
    (the spectrogram + the per-class recall bar chart). Deliberately NOT
    applied to confusion_matrix.png / accuracy_vs_snr.png -- those are
    written by src/evaluate.py, shared with the rest of the team (Colab
    notebook downloads, brief figures), so their style isn't this script's
    to change."""
    fig.patch.set_facecolor(_PANEL)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.xaxis.label.set_color(_TEXT_DIM)
        ax.yaxis.label.set_color(_TEXT_DIM)
        ax.title.set_color(_TEXT)
        ax.grid(color=_GRID, alpha=0.7, linewidth=0.6)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(_PANEL)
            leg.get_frame().set_edgecolor(_GRID)
            for txt in leg.get_texts():
                txt.set_color(_TEXT_DIM)


def _verdict_html(pred, tier, extra=""):
    color = _TIER_COLOR.get(tier, _TEXT_DIM)
    extra_html = (f'<div style="font-size:12px;color:{_TEXT_DIM};margin-top:6px;">{extra}</div>'
                  if extra else "")
    return f"""
    <div style="background:{_PANEL};border:1px solid {_GRID};border-radius:6px;
                padding:14px 16px;">
      <span style="display:inline-block;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
                   padding:2px 9px;border-radius:9px;background:{color}1a;color:{color};
                   font-weight:600;">{tier}</span>
      <div style="font-size:24px;font-weight:600;color:{_TEXT};margin-top:8px;
                  font-family:inherit;">{pred}</div>
      {extra_html}
    </div>
    """

TIER_OF = {}
for _cls in ["BPSK", "QPSK", "16QAM", "64QAM"]:
    TIER_OF[_cls] = "Civilian"
for _cls in ["LFM_RADAR", "FHSS"]:
    TIER_OF[_cls] = "Military"
TIER_OF["JAMMING"] = "Hostile"
TIER_OF["NOISE_FLOOR"] = "Empty"


def load_model():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN).to(DEVICE)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Train a model first "
            f"(python -m src.train), or copy your latest Colab checkpoint here."
        )
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    except RuntimeError as e:
        raise gr.Error(
            f"{CKPT_PATH} doesn't match the current model architecture "
            f"(stale checkpoint from before attention-pooling/NOISE_FLOOR were "
            f"added). Train a fresh one with the current code, or copy your "
            f"latest compatible checkpoint here. Details: {e}"
        )
    model.eval()
    return model


def load_iq_file(path, dtype=np.float32):
    """Same contract as src/infer.py: interleaved I,Q,I,Q,... """
    raw = np.fromfile(path, dtype=dtype)
    if raw.size % 2:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def make_spectrogram_figure(iq, title):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    t = np.arange(len(iq)) / FS * 1e6
    axes[0].plot(t, iq.real, lw=0.9, label="I", color=_TIER_COLOR["Civilian"])
    axes[0].plot(t, iq.imag, lw=0.9, alpha=0.85, label="Q", color=_TIER_COLOR["Military"])
    axes[0].set_title(title)
    axes[0].set_xlabel("time (µs)")
    axes[0].legend(fontsize=8)

    f_, t_, Z = stft(iq, fs=FS, nperseg=32, return_onesided=False)
    axes[1].pcolormesh(t_ * 1e6, np.fft.fftshift(f_) / 1e6,
                        np.fft.fftshift(np.abs(Z), axes=0), shading="gouraud", cmap="viridis")
    axes[1].set_xlabel("time (µs)")
    axes[1].set_ylabel("freq (MHz)")
    _style_light_axes(fig, axes)
    plt.tight_layout()
    return fig


def classify_iq(iq, model):
    if len(iq) < WINDOW_LEN:
        raise gr.Error(f"Input has {len(iq)} samples, need at least {WINDOW_LEN}.")
    iq = iq[:WINDOW_LEN]
    arr = preprocess_window(iq, WINDOW_LEN)
    x = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred = CLASSES[int(probs.argmax())]
    label_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    tier = TIER_OF.get(pred, "?")
    return iq, label_dict, _verdict_html(pred, tier)


def run_on_upload(file):
    if file is None:
        raise gr.Error("Upload a raw IQ file first (interleaved float32 I,Q,I,Q,...).")
    model = load_model()
    iq = load_iq_file(file.name if hasattr(file, "name") else file)
    iq, label_dict, verdict = classify_iq(iq, model)
    fig = make_spectrogram_figure(iq, "Uploaded sample")
    return fig, label_dict, verdict


def run_on_random_test_example():
    model = load_model()
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    _, _, test_idx = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    idx = int(np.random.choice(test_idx))
    arr = X[idx]  # already preprocessed (2, window_len)
    true_class = CLASSES[int(y[idx])]
    snr = snr_labels[idx]

    x = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred = CLASSES[int(probs.argmax())]
    label_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    iq_like = arr[0] + 1j * arr[1]
    fig = make_spectrogram_figure(
        iq_like, f"Real test example — true label: {true_class} @ {snr:.0f} dB")
    correct = "✓ correct" if pred == true_class else "✗ wrong"
    extra = f"{correct} — true label was {true_class} @ {snr:.0f} dB"
    verdict = _verdict_html(pred, TIER_OF.get(pred, "?"), extra)
    return fig, label_dict, verdict


def _tier_of_class(cls):
    for tier, members in TIERS.items():
        if cls in members:
            return tier
    return "?"


def build_dashboard(progress=gr.Progress()):
    """Re-runs the real evaluate.py against the CURRENT checkpoint, then reads
    back the fresh scorecard.json/PNGs it writes. Always live, never stale --
    the whole point of a dashboard is that it reflects whichever checkpoint is
    actually sitting in results/ right now, not a cached run from earlier."""
    if not CKPT_PATH.exists():
        raise gr.Error(f"No checkpoint at {CKPT_PATH}. Train or copy one there first.")

    progress(0.1, desc="Running evaluation on the held-out test split...")
    try:
        run_full_evaluation()
    except RuntimeError as e:
        raise gr.Error(f"Evaluation failed against the current checkpoint: {e}")

    progress(0.8, desc="Building charts...")
    with open(EVALS_DIR / "scorecard.json") as f:
        sc = json.load(f)

    per_class = sc["per_class"]
    bench = sc["benchmark"]
    coarse = sc["coarse_tier"]
    cvj = sc["comms_vs_jamming"]

    # --- summary markdown -------------------------------------------------
    lines = [f"### Benchmark: {'✅ PASS' if bench['passed'] else '❌ FAIL'} "
             f"(>{bench['benchmark_recall']:.0%} recall on judged classes)\n"]
    for cls, r in bench["judged_classes"].items():
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"- **{cls}**: recall {r['recall']:.1%} {mark}")
    if cvj:
        lines.append(f"\n**Comms vs Hostile CEMA** — discrimination accuracy "
                      f"{cvj['accuracy']:.1%}, jamming recall {cvj['jamming_recall']:.1%}, "
                      f"false alarm rate {cvj['false_alarm_rate']:.1%}")
    lines.append(f"\n**Coarse tier accuracy**: {coarse['accuracy']:.1%}")
    for tier, rec in coarse["per_tier_recall"].items():
        lines.append(f"  - {tier}: {rec:.1%}" if rec is not None else f"  - {tier}: n/a")
    summary_md = "\n".join(lines)

    # --- per-class recall bar chart, color-coded by tier -------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [c for c in CLASSES if per_class[c]["support"] > 0]
    recalls = [per_class[c]["recall"] for c in names]
    colors = [_TIER_COLOR[_tier_of_class(c)] for c in names]
    ax.bar(names, recalls, color=colors)
    ax.axhline(bench["benchmark_recall"], color="#e5484d", ls=":", lw=1.2,
               label=f"{bench['benchmark_recall']:.0%} benchmark")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall")
    leg = ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    _style_light_axes(fig, ax)
    leg.get_frame().set_facecolor(_PANEL)
    plt.tight_layout()

    cm_path = str(EVALS_DIR / "confusion_matrix.png")
    snr_path = str(EVALS_DIR / "accuracy_vs_snr.png")
    progress(1.0, desc="Done")
    return summary_md, fig, cm_path, snr_path


CUSTOM_CSS = f"""
:root {{
    --body-background-fill: {_BG};
    --background-fill-primary: {_PANEL};
    --background-fill-secondary: {_GRID};
    --border-color-primary: {_GRID};
    --block-background-fill: {_PANEL};
    --block-border-color: {_GRID};
    --block-label-background-fill: {_PANEL};
    --block-label-text-color: {_TEXT_DIM};
    --block-label-border-color: {_GRID};
    --block-title-text-color: {_TEXT};
    --body-text-color: {_TEXT};
    --body-text-color-subdued: {_TEXT_DIM};
    --input-background-fill: #ffffff;
    --input-border-color: {_GRID};
    --panel-background-fill: {_BG};
    --panel-border-color: {_GRID};
    --button-secondary-background-fill: #ffffff;
    --button-secondary-border-color: {_GRID};
    --button-secondary-text-color: {_TEXT};
    --button-primary-background-fill: {_TIER_COLOR["Civilian"]};
    --button-primary-background-fill-hover: {_TIER_COLOR["Civilian"]};
    --button-primary-text-color: #ffffff;
    --button-primary-border-color: {_TIER_COLOR["Civilian"]};
    --slider-color: {_TIER_COLOR["Civilian"]};
}}
body {{ background: {_BG}; }}
.gradio-container {{
    background: {_BG};
    font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
}}
#ow-header {{
    display:flex; align-items:baseline; justify-content:space-between;
    padding: 4px 2px 14px 2px; border-bottom: 1px solid {_GRID}; margin-bottom: 12px;
    flex-wrap: wrap; gap:8px;
}}
#ow-header .mark {{ font-size: 15px; letter-spacing: 0.2em; font-weight: 600; color: {_TEXT}; }}
#ow-header .sub {{ font-size: 11px; letter-spacing: 0.12em; color: {_TEXT_DIM}; }}
label span {{ letter-spacing: 0.04em; }}
footer {{ display:none !important; }}
"""

with gr.Blocks(title="Project Overwatch — Signal Classifier") as demo:
    gr.HTML(
        '<div id="ow-header">'
        '<span class="mark">PROJECT OVERWATCH</span>'
        '<span class="sub">TEAM NEXA &middot; RF / CEMA TRACK</span>'
        '</div>'
    )

    with gr.Tabs():
        with gr.Tab("Classify a signal"):
            gr.Markdown(
                "Real inference from the trained model, not a simulation. "
                "Upload a raw IQ file, or try a real held-out test example."
            )
            with gr.Row():
                with gr.Column():
                    file_in = gr.File(label="Raw IQ file (interleaved float32 I,Q,I,Q,...)")
                    run_btn = gr.Button("Classify uploaded file", variant="primary")
                    random_btn = gr.Button("Try a random real test example instead")
                    verdict_out = gr.HTML()
                    label_out = gr.Label(label="Class probabilities", num_top_classes=8)
                with gr.Column():
                    plot_out = gr.Plot(label="Signal (amplitude + spectrogram)")

            run_btn.click(run_on_upload, inputs=file_in, outputs=[plot_out, label_out, verdict_out])
            random_btn.click(run_on_random_test_example, inputs=None,
                              outputs=[plot_out, label_out, verdict_out])

        with gr.Tab("Results dashboard"):
            gr.Markdown(
                "Runs the real evaluation pipeline against whichever checkpoint is "
                "currently at `results/best_model.pt` — always live, never a stale cached run."
            )
            dash_btn = gr.Button("Run evaluation", variant="primary")
            dash_summary = gr.Markdown()
            with gr.Row():
                dash_bar = gr.Plot(label="Per-class recall")
                dash_cm = gr.Image(label="Confusion matrix", show_label=True)
            dash_snr = gr.Image(label="Accuracy vs SNR", show_label=True)

            dash_btn.click(build_dashboard, inputs=None,
                            outputs=[dash_summary, dash_bar, dash_cm, dash_snr])

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"))
