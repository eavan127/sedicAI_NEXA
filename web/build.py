"""Generates web/index.html from web/index.template.html.

Two things happen here, both so the published site is a self-contained
folder with no build step of its own:

  1. assets/sedic_logo.png is inlined as a data: URI. The asset is
     gitignored (see src/ui/app.py:_logo_path for the same problem on the
     Gradio side), so it cannot be fetched as a file by a static host.
  2. The ONNX checkpoints are copied into web/models/ as SINGLE files --
     torch's exporter writes weights to a sidecar .onnx.data, which
     onnxruntime-web's WASM backend cannot load ("Module.MountedFiles is
     not available").

Usage:
    python web/build.py
"""
import base64
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
MODELS_OUT = WEB / "models"
ONNX_SRC = REPO / "results" / "onnx"
CHECKPOINTS = ["best_model", "ensemble_0", "ensemble_1", "ensemble_2",
                "ensemble_3", "ensemble_4"]


def build_html():
    template = (WEB / "index.template.html").read_text(encoding="utf-8")
    logo = REPO / "assets" / "sedic_logo.png"
    if logo.is_file():
        b64 = base64.b64encode(logo.read_bytes()).decode()
        src = f"data:image/png;base64,{b64}"
    else:
        print(f"  warning: {logo} not found — header will show a broken image")
        src = ""
    out = template.replace("__LOGO_BASE64__", src)
    (WEB / "index.html").write_text(out, encoding="utf-8")
    print(f"  index.html  ({len(out) / 1024:.0f} KB)")


def build_models():
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINTS:
        src = ONNX_SRC / f"{name}.onnx"
        if not src.is_file():
            print(f"  warning: {src} missing — run scripts/export_onnx.py first")
            continue
        model = onnx.load(str(src))          # pulls in the sidecar .data
        dst = MODELS_OUT / f"{name}.onnx"
        onnx.save_model(model, str(dst), save_as_external_data=False)
        print(f"  models/{name}.onnx  ({dst.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    print("building web/")
    build_html()
    build_models()
    print("done")
