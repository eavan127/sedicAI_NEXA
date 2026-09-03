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


def _f32(path, arr):
    """Raw little-endian float32, the layout Float32Array reads directly."""
    import numpy as np
    np.ascontiguousarray(arr, dtype="<f4").tofile(path)
    return path.stat().st_size


def build_data():
    """Exports the arrays the static pages need, which a static host cannot
    read from data/processed at request time.

    Small on purpose: the civilian library is a fixed slice (<=400 windows
    per class) and the test split is whatever data/processed holds. Both are
    a few hundred KB, not the 25 GB of data/raw.
    """
    import json

    import numpy as np

    from src.config import CFG, CLASSES
    from src.measure import (C42_BOUNDARY, C42_MIN_SNR_DB,
                              C42_POOLED_ACCURACY,
                              MIN_WINDOWS_FOR_C42_DECISION)
    from src.train import load_data, stratified_split
    from src.ui.session import CLEANEST_LIBRARY_SNR_DB, civilian_library

    data_out = WEB / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    # --- civilian library: restores the 4 civilian scenario cases ----------
    lib = civilian_library()
    manifest = {"snr_db": CLEANEST_LIBRARY_SNR_DB, "window_len": CFG["signal"]["window_len"],
                 "classes": {}}
    for cls, windows in lib.items():
        name = f"civilian_{cls}.bin"
        size = _f32(data_out / name, windows)
        manifest["classes"][cls] = {"file": name, "n": int(windows.shape[0])}
        print(f"  data/{name}  ({size / 1024:.0f} KB, {windows.shape[0]} windows)")
    (data_out / "civilian_library.json").write_text(json.dumps(manifest))

    # --- test split ---------------------------------------------------------
    # Loaded here for the breakdown below, but deliberately NOT exported. An
    # earlier version shipped X[test] as a .bin so the browser could run the
    # model over it; that design was dropped because
    # src/ui/pages/performance.py is explicit that the page must DISPLAY what
    # the Python evaluation produced and never recompute a metric. The
    # breakdown is therefore computed below, at build time, and only its
    # results are shipped -- which also keeps 77 MB of raw IQ out of the
    # repo and off the static host.
    X, y, snr = load_data()
    d = CFG["dataset"]
    _, _, test = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])

    # --- Performance page ---------------------------------------------------
    # src/ui/pages/performance.py is explicit that the page DISPLAYS what the
    # Python evaluation produced and never recomputes a metric, "a worse
    # failure than the page not existing". So the numbers are computed HERE,
    # by the same src/breakdown.py the Gradio page calls, and the browser only
    # renders them.
    #
    # Stamped with provenance because data/processed can hold either a smoke
    # run or the real dataset and the two look identical otherwise -- the page
    # prints this so a reader can never mistake one for the other.
    import datetime

    from src.breakdown import single_vs_multi
    from src.ui.app_models import load_model, model_label

    model = load_model("auto")
    breakdown = single_vs_multi(model, X[test], y[test], snr[test],
                                 classes=list(CLASSES))
    scorecard_path = REPO / "evals" / "scorecard.json"
    scorecard = (json.loads(scorecard_path.read_text())
                  if scorecard_path.is_file() else None)
    # The ensemble's own judged-class figures, which is what the team
    # actually submits. Written by scripts/train_ensemble.py, and a DIFFERENT
    # shape from scorecard.json (judged classes only, no per-class
    # precision/f1/support table).
    ens_path = REPO / "evals" / "ensemble_scorecard.json"
    ensemble_scorecard = (json.loads(ens_path.read_text())
                           if ens_path.is_file() else None)
    (data_out / "performance.json").write_text(json.dumps({
        # Two DIFFERENT models are represented on this page and conflating
        # them would be a provenance error of exactly the kind
        # src/ui/pages/performance.py exists to prevent:
        #   - the scorecard table comes from src.evaluate, which defaults to
        #     the SINGLE checkpoint (best_model.pt) unless run --ensemble,
        #     and records nothing in the file about which it was;
        #   - the breakdown chart is computed above with load_model("auto"),
        #     i.e. the ensemble whenever all five members are present.
        "scorecard_source": "evals/scorecard.json — src.evaluate (single checkpoint unless run with --ensemble)",
        "breakdown_model": model_label("auto"),
        "ensemble_scorecard": ensemble_scorecard,
        "model_label": model_label("auto"),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "total_windows": int(X.shape[0]),
            "test_windows": int(len(test)),
            "val_frac": d["val_frac"], "test_frac": d["test_frac"],
            "seed": d["seed"],
        },
        "snr_bins": CFG["snr_bins_db"],
        "classes": list(CLASSES),
        "judged_classes": CFG["judged_classes"],
        "benchmark_recall": CFG["benchmark_recall"],
        "breakdown": {
            "recall": breakdown.recall, "totals": breakdown.totals,
            "support": breakdown.support, "n_windows": breakdown.n_windows,
        },
        "scorecard": scorecard,
        "scorecard_mtime": (datetime.datetime.fromtimestamp(
            scorecard_path.stat().st_mtime).isoformat(timespec="seconds")
            if scorecard_path.is_file() else None),
    }))
    print(f"  data/performance.json  ({len(test)} test windows, "
          f"{breakdown.n_windows} single/multi)")

    # --- constants + model card -------------------------------------------
    from src.models.amc_cnn import AMC_CNN
    import torch
    model = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"])
    model.load_state_dict(torch.load(REPO / "results" / "best_model.pt", map_location="cpu"))
    card = {
        "architecture": "EnsembleModel",
        "member_architecture": type(model).__name__,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "branches": {name: int(sum(p.numel() for p in getattr(model, name).parameters()))
                      for name in ("iq_branch", "stft_branch")},
        "classes": list(CLASSES),
        "window_len": CFG["signal"]["window_len"],
        "fs": CFG["signal"]["fs"],
        "thresholds": CFG.get("multilabel_thresholds_per_class", {}),
        "benchmark_recall": CFG["benchmark_recall"],
        "judged_classes": CFG["judged_classes"],
        "c42": {
            "boundary": C42_BOUNDARY,
            "min_windows": MIN_WINDOWS_FOR_C42_DECISION,
            "min_snr_db": C42_MIN_SNR_DB,
            "pooled_accuracy": {str(k): v for k, v in C42_POOLED_ACCURACY.items()},
        },
    }
    (data_out / "model_card.json").write_text(json.dumps(card, indent=1))
    print(f"  data/model_card.json  ({card['parameters']:,} params)")


if __name__ == "__main__":
    print("building web/")
    build_html()
    build_models()
    build_data()
    print("done")
