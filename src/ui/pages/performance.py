"""Performance -- the existing Results dashboard, ported unchanged.

This page DISPLAYS what src/evaluate.py produces. It never recomputes a
metric. If it derived its own recall or false-alarm rate, a judge could be
shown numbers that disagree with the official scorecard -- a worse failure
than the page not existing.

Numbers here are ALWAYS per-window, ungated and unsmoothed. The RF Replay
smoothing toggle, the NOISE_FLOOR gate and the event hold do not reach this
page, by construction: nothing here imports them.

Deliberately keeps the LIGHT palette. confusion_matrix.png and
accuracy_vs_snr.png are written by src/evaluate.py and shared with the rest of
the team (Colab downloads, brief figures); restyling them is not this
package's call. The mismatch with the dark console is intentional and
documented in the spec.
"""
import json

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CFG, CLASSES, REPO_ROOT, TIERS

CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
EVALS_DIR = REPO_ROOT / CFG["paths"]["evals"]

# Light palette, matching the dashboard this was ported from.
_PANEL = "#ffffff"
_GRID = "#e1e7ee"
_TEXT = "#17212b"
_TEXT_DIM = "#5b6b7c"
_TIER_COLOR = {"Civilian": "#0d9488", "Military": "#b45309",
               "Hostile": "#dc2626", "Empty": "#5b6b7c"}
_TIER_OF = {cls: tier for tier, members in TIERS.items() for cls in members}


def _style_light_axes(fig, axes):
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


def _build_dashboard(progress=gr.Progress()):
    """Re-runs the real evaluate.py against the CURRENT checkpoint, then reads
    back the fresh scorecard.json/PNGs it writes. Always live, never stale --
    the whole point of a dashboard is that it reflects whichever checkpoint is
    actually sitting in results/ right now, not a cached run from earlier."""
    # Imported here, not at module scope: src.evaluate pulls in sklearn and
    # runs a full evaluation's worth of imports, and this page is one of six.
    from src.evaluate import evaluate as run_full_evaluation

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

    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [c for c in CLASSES if per_class[c]["support"] > 0]
    recalls = [per_class[c]["recall"] for c in names]
    colors = [_TIER_COLOR[_TIER_OF[c]] for c in names]
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

    progress(1.0, desc="Done")
    return (summary_md, fig, str(EVALS_DIR / "confusion_matrix.png"),
            str(EVALS_DIR / "accuracy_vs_snr.png"))


def build():
    gr.Markdown("### Performance")
    gr.Markdown(
        "Runs the real evaluation pipeline against whichever checkpoint is at "
        "`results/best_model.pt`. Always per-window, ungated and unsmoothed — "
        "the RF Replay display settings have no effect here."
    )
    run = gr.Button("Run evaluation", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        bar = gr.Plot(label="Per-class recall")
        cm = gr.Image(label="Confusion matrix")
    snr = gr.Image(label="Accuracy vs SNR")

    run.click(_build_dashboard, inputs=None, outputs=[summary, bar, cm, snr])
