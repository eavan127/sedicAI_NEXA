"""Assembles the six OMNI pages."""
import gradio as gr
import torch

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.ui.pages import (alerts, model_page, overview, performance, rf_replay,
                           signal_analysis)
from src.ui.palette import BG, PANEL, TEXT, TEXT_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"

CUSTOM_CSS = f"""
.gradio-container {{ background: {BG} !important; }}
.gradio-container, .gradio-container * {{ color: {TEXT}; }}
.gr-panel, .block, .form {{ background: {PANEL} !important; border-color: #1f2933 !important; }}
footer {{ display: none !important; }}
"""


def load_model():
    """Read from disk on every call -- no caching, so dropping a freshly
    trained checkpoint into results/ takes effect without a restart."""
    model = AMC_CNN(num_classes=len(CLASSES),
                     input_len=CFG["signal"]["window_len"]).to(DEVICE)
    if not CKPT_PATH.exists():
        raise gr.Error(f"No checkpoint at {CKPT_PATH}. Train one first.")
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    except RuntimeError as exc:
        raise gr.Error(
            f"{CKPT_PATH} does not match the current model architecture. "
            f"Train a fresh one with the current code. Details: {exc}")
    model.eval()
    return model


THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate")


def build_app():
    with gr.Blocks(title="OMNI — RF Spectrum Intelligence") as demo:
        gr.HTML(
            f'<div style="padding:14px 0 6px 0;">'
            f'<div style="font-size:26px;font-weight:700;letter-spacing:0.14em;'
            f'color:{TEXT};">OMNI</div>'
            f'<div style="font-size:12px;color:{TEXT_DIM};letter-spacing:0.04em;">'
            f'AI-Powered RF Spectrum Intelligence &nbsp;·&nbsp; TEAM NEXA '
            f'&nbsp;·&nbsp; BASEBAND · fs 3.2 MHz &nbsp;·&nbsp; '
            f'● REPLAY — no live SDR</div></div>')

        state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview.build(state)
            with gr.Tab("RF Replay"):
                rf_replay.build(state, load_model)
            with gr.Tab("Signal Analysis"):
                signal_analysis.build(state)
            with gr.Tab("Performance"):
                performance.build()
            with gr.Tab("Model"):
                model_page.build(load_model)
            with gr.Tab("Alerts"):
                alerts.build(state)

    return demo


def launch(**kwargs):
    """Gradio 6 takes css and theme on launch(), not on the Blocks
    constructor, so they are applied here rather than at build time."""
    return build_app().launch(css=CUSTOM_CSS, theme=THEME, **kwargs)
