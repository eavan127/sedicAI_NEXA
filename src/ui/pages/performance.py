"""Performance -- the existing Results dashboard, ported unchanged.

This page DISPLAYS what src/evaluate.py produces. It never recomputes a
metric. If it derived its own recall or false-alarm rate, a judge could be
shown numbers that disagree with the official scorecard -- a worse failure
than the page not existing.

Numbers here are ALWAYS per-window, ungated and unsmoothed. The RF Replay
smoothing toggle, the NOISE_FLOOR gate and the event hold do not reach this
page, by construction: nothing here imports them.

confusion_matrix.png and accuracy_vs_snr.png are written by src/evaluate.py
and shared with the rest of the team (Colab downloads, brief figures);
restyling them is not this package's call. They are embedded as-is. The bar
chart this file draws itself does follow the console palette.
"""
import json

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CFG, CLASSES, REPO_ROOT, TIERS
from src.ui.palette import GRID, MPL_FONT, PANEL, TEXT, TEXT_DIM, TIER_COLOR

CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
EVALS_DIR = REPO_ROOT / CFG["paths"]["evals"]

# Shares the console palette now that the console is light too -- there is no
# longer a reason for this page to carry its own colour table. The FIGURES
# written by src/evaluate.py are still not ours to restyle; only the bar chart
# this file draws itself is affected.
_PANEL, _GRID, _TEXT, _TEXT_DIM = PANEL, GRID, TEXT, TEXT_DIM
_TIER_COLOR = TIER_COLOR
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = MPL_FONT
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

    # Evaluate whatever the console itself loads. This previously called
    # evaluate() with no arguments, which always scores the SINGLE
    # best_model.pt -- so with an ensemble present the Performance page
    # reported one model while RF Replay ran five, and the page could show
    # FAIL for a configuration that passes. The console must not disagree
    # with itself about its own results.
    from src.ui.app_models import ensemble_available, model_label
    use_ensemble = ensemble_available()
    progress(0.1, desc=f"Evaluating {model_label('auto')}...")
    try:
        run_full_evaluation(ensemble=use_ensemble, n_models=5)
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
    # --- category view -------------------------------------------------
    # Every class, grouped by tier. The benchmark judges three classes, so a
    # page showing only those reads as though the model knows three things.
    # Civilian is not judged, but "can it tell traffic from interference" is
    # exactly what the CEMA criterion turns on, so it belongs on screen.
    lines.append("\n#### By category\n")
    lines.append("| category | classes | tier recall |")
    lines.append("|---|---|---|")
    for tier, members in TIERS.items():
        present = [c for c in members if per_class.get(c, {}).get("support", 0)]
        if not present:
            continue
        rec = coarse["per_tier_recall"].get(tier)
        detail = ", ".join(f"{c} {per_class[c]['recall']:.0%}" for c in present)
        lines.append(f"| **{tier}** | {detail} | "
                      + ("—" if rec is None else f"**{rec:.1%}**") + " |")
    if cvj:
        lines.append(f"| **CEMA** — comms vs hostile | jamming recall "
                      f"{cvj['jamming_recall']:.1%}, false alarm "
                      f"{cvj['false_alarm_rate']:.2%} | "
                      f"**{cvj['accuracy']:.1%}** |")

    lines.append(f"\n**Coarse tier accuracy**: {coarse['accuracy']:.1%}"
                  f" &nbsp;·&nbsp; evaluated: **{model_label('auto')}**")
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


def _build_breakdown(progress=gr.Progress()):
    """Single- vs multi-signal recall across the SNR sweep.

    Additive to the scorecard, not a competing one: same test split, same
    per-class thresholds, same checkpoint. Ungated and unsmoothed like
    everything else on this page.
    """
    from src.breakdown import single_vs_multi
    from src.train import load_data, stratified_split
    from src.ui.app_models import load_model, model_label

    progress(0.1, desc="Loading test split...")
    X, y, snr = load_data()
    d = CFG["dataset"]
    _, _, test = stratified_split(y, snr, d["val_frac"], d["test_frac"],
                                   d["seed"])

    progress(0.3, desc="Running the model over the test split...")
    # ALL eight classes, not just the judged three. Civilian classes are not
    # scored by the benchmark, but "can it tell traffic from interference"
    # is the question the CEMA criterion actually turns on, and leaving them
    # out made the page look like the model only knows three things.
    r = single_vs_multi(load_model("auto"), X[test], y[test], snr[test],
                         classes=list(CLASSES))

    progress(0.85, desc="Building chart...")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cls in r.classes:
        colour = _TIER_COLOR[_TIER_OF[cls]]
        for group, style, marker in (("single", "-", "o"), ("multi", "--", "s")):
            xs = [s for s in r.snr_bins if r.recall[group][cls][s] is not None]
            ys = [r.recall[group][cls][s] for s in xs]
            if xs:
                ax.plot(xs, ys, style, marker=marker, color=colour,
                         linewidth=1.6, markersize=4,
                         label=f"{cls} — {group}")
    ax.axhline(CFG["benchmark_recall"] * 100, color="#e5484d", ls=":", lw=1.2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("recall (%)")
    ax.set_ylim(0, 102)
    leg = ax.legend(fontsize=6, ncol=4, loc="lower right")
    _style_light_axes(fig, ax)
    leg.get_frame().set_facecolor(_PANEL)
    plt.tight_layout()

    head = (f"**{model_label('auto')}** &nbsp;·&nbsp; "
            f"{r.n_windows['single']:,} single-signal / "
            f"{r.n_windows['multi']:,} multi-signal windows in the test split\n\n"
            f"Solid = one emitter in the window. Dashed = emitters overlapping. "
            f"Dotted red line is the {CFG['benchmark_recall']:.0%} gate.\n\n")
    head += ("| category | class | " + " | ".join(f"{s:+d} dB" for s in r.snr_bins)
              + " | all |\n")
    head += "|" + "---|" * (len(r.snr_bins) + 3) + "\n"
    for group in ("single", "multi"):
        for cls in r.classes:
            # A class absent from this group entirely (NOISE_FLOOR never
            # co-occurs, so it has no multi-signal rows) would otherwise print
            # a row of em-dashes that reads like a failure rather than an
            # absence.
            if all(r.recall[group][cls][sb] is None for sb in r.snr_bins):
                continue
            cells = []
            for sb in r.snr_bins:
                v = r.recall[group][cls][sb]
                cells.append("—" if v is None else f"{v:.0f}%")
            tot = r.totals[group][cls]
            head += (f"| {_TIER_OF[cls]} | {cls} ({group}) | "
                      + " | ".join(cells) + " | "
                      + ("—" if tot is None else f"**{tot:.0f}%**") + " |\n")

    progress(1.0, desc="Done")
    return head, fig


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

    gr.Markdown("---")
    gr.Markdown("#### Single-signal vs multi-signal, across SNR")
    gr.Markdown(
        "The headline scorecard averages two different regimes together: "
        "windows carrying one emitter, and windows where emitters overlap. "
        "Same test split, same per-class thresholds, same checkpoint."
    )
    bd_run = gr.Button("Run breakdown", variant="primary")
    bd_summary = gr.Markdown()
    bd_plot = gr.Plot(label="Recall vs SNR — solid: single signal, dashed: overlapping")
    bd_run.click(_build_breakdown, inputs=None, outputs=[bd_summary, bd_plot])
