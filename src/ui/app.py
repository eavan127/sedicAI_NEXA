"""Assembles the six OMNI pages."""
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN
from src.ui.app_models import load_model, model_label  # noqa: F401
from src.ui.pages import (model_page, performance, rf_replay,
                           signal_analysis)
from src.ui.palette import (BG, BRAND_OLIVE, BRAND_OLIVE_DARK,
                             BRAND_OLIVE_TINT, BRAND_SLATE, FONT_STACK, GRID,
                             MONO_STACK, PANEL, TEXT, TEXT_DIM)

CUSTOM_CSS = f"""
/* Gradio resolves its own palette from CSS custom properties, and picks the
   DARK set when the browser reports a dark colour scheme. Setting only
   backgrounds left --body-text-color at #f1f5f9 -- near-white text on a white
   panel, contrast ratio 1.03, invisible. Overriding the variables themselves
   fixes every component at once; overriding `color` per element does not,
   because Gradio's own scoped classes are more specific.
   The .dark block repeats them so a dark-mode browser gets the light theme
   too, rather than half of each. */
.gradio-container, .gradio-container .dark, .dark {{
  --body-background-fill: {BG};
  --body-text-color: {TEXT};
  --body-text-color-subdued: {TEXT_DIM};
  --background-fill-primary: {PANEL};
  --background-fill-secondary: {BG};
  --block-background-fill: {PANEL};
  --panel-background-fill: {PANEL};
  --block-label-background-fill: {PANEL};
  --block-label-text-color: {TEXT_DIM};
  --block-title-text-color: {TEXT};
  --block-info-text-color: {TEXT_DIM};
  --block-border-color: {GRID};
  --border-color-primary: {GRID};
  --border-color-accent: {BRAND_OLIVE};
  --input-background-fill: {PANEL};
  --input-border-color: {GRID};
  --input-placeholder-color: {TEXT_DIM};
  --table-even-background-fill: {PANEL};
  --table-odd-background-fill: {BG};
  --table-border-color: {GRID};
  /* Gradio reads --table-row-focus for the hovered row
     (.virtual-row:hover in its own stylesheet). The name used here before was
     --table-row-focus-background-fill, which does not exist in Gradio 6 and
     therefore never applied -- the row fell back to Gradio's default dark
     navy, and with body cells that had no surface of their own that navy
     showed through under this file's dark ink. A light tint keeps the hovered
     row readable with the same dark text as every other row, instead of
     needing a white-text override that would also repaint the tier colours. */
  --table-row-focus: {BRAND_OLIVE_TINT};
  --button-secondary-background-fill: {PANEL};
  --button-secondary-text-color: {TEXT};
  --button-secondary-border-color: {GRID};
  --link-text-color: {BRAND_OLIVE_DARK};
  --color-accent: {BRAND_OLIVE};
  --color-accent-soft: {BRAND_OLIVE_TINT};
}}
.gradio-container {{
  background: {BG} !important;
  font-family: {FONT_STACK} !important;
  /* Gradio caps the shell at ~1200px. The waterfall and the detections table
     both want more than that, and the cap forced a horizontal scroll that made
     the table unreadable without zooming. */
  max-width: 100% !important;
  width: 100% !important;
  padding-left: 26px !important;
  padding-right: 26px !important;
}}
.gradio-container *, .gradio-container p, .gradio-container span,
.gradio-container label, .gradio-container h1, .gradio-container h2,
.gradio-container h3, .gradio-container td, .gradio-container th {{
  font-family: {FONT_STACK};
}}
button.primary, .gr-button-primary {{
  background: {BRAND_OLIVE} !important;
  border-color: {BRAND_OLIVE_DARK} !important;
  color: #ffffff !important;
}}
button.primary:hover, .gr-button-primary:hover {{
  background: {BRAND_OLIVE_DARK} !important;
}}
.tab-nav button {{ color: {TEXT_DIM} !important; }}
.tab-nav button.selected {{
  color: {BRAND_OLIVE_DARK} !important;
  border-bottom-color: {BRAND_OLIVE} !important;
  font-weight: 600;
}}
thead th {{ background: {BRAND_OLIVE_TINT} !important; color: {TEXT} !important; }}
/* Inline code keeps Gradio's dark code surface even after the variables are
   overridden, which put dark text on a dark chip. Give it the brand tint. */
.gradio-container code, .gradio-container kbd, .gradio-container samp {{
  background: {BRAND_OLIVE_TINT} !important;
  color: {BRAND_OLIVE_DARK} !important;
  font-family: {MONO_STACK};
  padding: 1px 5px;
  border-radius: 3px;
}}
footer {{ display: none !important; }}
/* No italics anywhere. Gradio italicises markdown emphasis and some captions
   by default; killing it globally is more reliable than auditing every
   string. Emphasis is carried by weight and colour instead. */
.gradio-container em, .gradio-container i, .gradio-container * {{
  font-style: normal !important;
}}
/* The detections table's BODY cells had no background of their own -- only
   thead was given one -- so they computed to rgba(0,0,0,0) and inherited
   whatever surface sat behind them, which put this file's dark ink (#121C27)
   on a dark ground and made the table unreadable.

   Targeting Gradio 6's own class names rather than guessing: it stripes with
   .row-odd (not nth-child, which an earlier attempt used and which fights it),
   and marks selection with .body-cell.cell-selected -- which sets only a
   box-shadow ring, never a background. So the dark row was never Gradio's
   selection styling; it was cells with no surface of their own showing the
   ground through. Every state below therefore sets background AND colour as a
   pair, so no combination can leave dark text on a dark surface. */
.gradio-container table tbody td,
.gradio-container table tbody th,
.gradio-container table tbody td .cell-wrap {{
  background-color: {PANEL} !important;
  color: {TEXT} !important;
}}
.gradio-container table tbody tr.row-odd td,
.gradio-container table tbody tr.row-odd td .cell-wrap {{
  background-color: {BG} !important;
  color: {TEXT} !important;
}}
/* Selection keeps Gradio's ring; we only guarantee the surface under it. */
.gradio-container table tbody td.cell-selected,
.gradio-container table tbody td.cell-selected .cell-wrap {{
  background-color: {BRAND_OLIVE_TINT} !important;
  color: {BRAND_OLIVE_DARK} !important;
}}
/* A transparent scroll container shows the host surface through the gap
   beside a short table. */
.gradio-container .table-wrap,
.gradio-container [class*="dataframe"] {{
  background-color: {PANEL} !important;
}}
/* Hover and selection, stated as a background+colour PAIR on every element
   that paints: the row, the cell, and the .cell-wrap div the text actually
   lives in. Setting only one of them is what produced dark-on-dark before. */
.gradio-container table tbody tr:hover td,
.gradio-container table tbody tr:hover td .cell-wrap,
.gradio-container tr.virtual-row:hover td,
.gradio-container tr.virtual-row:hover td .cell-wrap {{
  background-color: {BRAND_OLIVE_TINT} !important;
  color: {TEXT} !important;
}}
"""

LOGO_NAME = "sedic_logo.png"


def _main_worktree_root():
    """Root of the MAIN checkout, as seen from a linked git worktree.

    In a linked worktree `.git` is a FILE containing `gitdir: <path>`, where
    <path> is `<main>/.git/worktrees/<name>`. Two levels up from that is
    `<main>/.git`, whose parent is the main checkout. Returns None when this
    is the main checkout itself (where `.git` is a directory), or when the
    file is missing or malformed -- callers treat None as "no extra place to
    look", never as an error.
    """
    dotgit = REPO_ROOT / ".git"
    if not dotgit.is_file():
        return None
    try:
        line = dotgit.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir:"):
            return None
        return Path(line.split(":", 1)[1].strip()).parents[1].parent
    except (OSError, IndexError, ValueError):
        return None


def _logo_path():
    """Locate the logo, which is gitignored and so exists in only one checkout.

    REPO_ROOT is derived from this file's location, so running the app from a
    linked worktree pointed it at `<worktree>/assets/` -- a directory that does
    not exist there, because an ignored file is never materialised into a
    worktree. The logo silently fell back to the typographic lockup for anyone
    not running from the main checkout.

    Candidates, in order: an explicit SEDIC_LOGO override, this checkout, then
    the main checkout. Returns None if none exist, and the caller falls back to
    type as before.
    """
    override = os.environ.get("SEDIC_LOGO")
    candidates = [Path(override)] if override else []
    candidates.append(REPO_ROOT / "assets" / LOGO_NAME)

    main_root = _main_worktree_root()
    if main_root is not None:
        candidates.append(main_root / "assets" / LOGO_NAME)

    return next((p for p in candidates if p.is_file()), None)


def _logo_html():
    """SEDIC 26 logo, if the asset can be found -- see _logo_path().

    Falls back to a typographic lockup rather than a broken image, so the app
    still runs for anyone who has not copied the asset in -- it is not checked
    into git.
    """
    path = _logo_path()
    if path is not None:
        import base64
        b64 = base64.b64encode(path.read_bytes()).decode()
        return (f'<img src="data:image/png;base64,{b64}" alt="SEDIC 26" '
                f'style="height:52px;width:auto;display:block;">')
    return (f'<div style="font-size:26px;font-weight:800;letter-spacing:0.08em;'
            f'color:{BRAND_OLIVE};line-height:1;">SEDIC<span '
            f'style="font-size:16px;vertical-align:super;">26</span></div>')


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
            f'<div style="font-size:13px;color:{TEXT_DIM};letter-spacing:0.02em;'
            f'margin-top:4px;">AI-Powered RF Spectrum Intelligence</div>'
            f'</div></div>')

        state = gr.State(None)

        with gr.Row():
            # Navigation Sidebar
            with gr.Column(scale=1, min_width=180):
                gr.Markdown("### Menu")
                nav_buttons = []
                pages = [
                    ("RF Replay", rf_replay.build, (state, load_model)),
                    ("Signal Analysis", signal_analysis.build, (state,)),
                    ("Performance", performance.build, ()),
                    ("Model", model_page.build, (load_model,))
                ]
                
                for idx, (name, _, _) in enumerate(pages):
                    btn = gr.Button(name, variant="primary" if idx == 0 else "secondary")
                    nav_buttons.append(btn)
            
            # Main Content Area
            with gr.Column(scale=5):
                page_containers = []
                for idx, (_, build_fn, args) in enumerate(pages):
                    with gr.Column(visible=(idx == 0)) as page:
                        build_fn(*args)
                    page_containers.append(page)
        
        def make_show_page(idx):
            def show():
                p_updates = [gr.update(visible=(i == idx)) for i in range(len(page_containers))]
                b_updates = [gr.update(variant="primary" if i == idx else "secondary") for i in range(len(nav_buttons))]
                return p_updates + b_updates
            return show
            
        for idx, btn in enumerate(nav_buttons):
            btn.click(
                fn=make_show_page(idx),
                inputs=None,
                outputs=page_containers + nav_buttons
            )

    return demo


def launch(**kwargs):
    """Gradio 6 takes css and theme on launch(), not on the Blocks
    constructor, so they are applied here rather than at build time."""
    return build_app().launch(css=CUSTOM_CSS, theme=THEME, **kwargs)
