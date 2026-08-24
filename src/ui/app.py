"""Assembles the six OMNI pages."""
import gradio as gr
import torch

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.ui.pages import (alerts, model_page, overview, performance, rf_replay,
                           signal_analysis)
from src.ui.palette import (BG, BRAND_OLIVE, BRAND_OLIVE_DARK,
                             BRAND_OLIVE_TINT, BRAND_SLATE, FONT_STACK, GRID,
                             PANEL, TEXT, TEXT_DIM)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"

LOGO_PATH = REPO_ROOT / "assets" / "sedic_logo.png"

CUSTOM_CSS = f"""
.gradio-container {{
  background: {BG} !important;
  font-family: {FONT_STACK} !important;
}}
.gradio-container, .gradio-container * {{
  color: {TEXT};
  font-family: {FONT_STACK};
}}
.gr-panel, .block, .form {{
  background: {PANEL} !important;
  border-color: {GRID} !important;
}}
button.primary, .gr-button-primary {{
  background: {BRAND_OLIVE} !important;
  border-color: {BRAND_OLIVE_DARK} !important;
  color: #ffffff !important;
}}
button.primary:hover, .gr-button-primary:hover {{
  background: {BRAND_OLIVE_DARK} !important;
}}
.tab-nav button.selected {{
  color: {BRAND_OLIVE_DARK} !important;
  border-bottom-color: {BRAND_OLIVE} !important;
  font-weight: 600;
}}
thead th {{ background: {BRAND_OLIVE_TINT} !important; }}
footer {{ display: none !important; }}
"""


def _logo_html():
    """SEDIC 26 logo, if the file has been placed in assets/.

    Falls back to a typographic lockup rather than a broken image, so the app
    still runs for anyone who has not copied the asset in -- it is not checked
    into git.
    """
    if LOGO_PATH.exists():
        import base64
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return (f'<img src="data:image/png;base64,{b64}" alt="SEDIC 26" '
                f'style="height:52px;width:auto;display:block;">')
    return (f'<div style="font-size:26px;font-weight:800;letter-spacing:0.08em;'
            f'color:{BRAND_OLIVE};line-height:1;">SEDIC<span '
            f'style="font-size:16px;vertical-align:super;">26</span></div>')


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
            f'<div style="display:flex;align-items:center;gap:22px;'
            f'padding:16px 0 12px 0;border-bottom:2px solid {BRAND_OLIVE};'
            f'margin-bottom:14px;">'
            + _logo_html() +
            f'<div style="border-left:1px solid {GRID};padding-left:22px;">'
            f'<div style="font-size:24px;font-weight:700;letter-spacing:0.16em;'
            f'color:{BRAND_SLATE};line-height:1.1;">OMNI</div>'
            f'<div style="font-size:12px;color:{TEXT_DIM};letter-spacing:0.03em;'
            f'margin-top:3px;">AI-Powered RF Spectrum Intelligence '
            f'&nbsp;·&nbsp; TEAM NEXA</div>'
            f'<div style="font-size:11px;color:{BRAND_OLIVE};font-weight:600;'
            f'letter-spacing:0.06em;margin-top:4px;">BASEBAND · fs 3.2 MHz '
            f'&nbsp;·&nbsp; ● REPLAY — no live SDR</div></div></div>')

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
