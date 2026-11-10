"""Generates web/index.html from web/index.template.html.

Two things happen here, both so the published site is a self-contained
folder with no build step of its own:

  1. assets/sedic_logo.png is inlined as a data: URI. The asset is
     gitignored (see src/ui/app.py:_logo_path for the same problem on the
     Gradio side), so it cannot be fetched as a file by a static host.
  2. The browser model is copied into web/models/best_model.onnx as a
     SINGLE file -- torch's exporter writes weights to a sidecar .onnx.data,
     which onnxruntime-web's WASM backend cannot load ("Module.MountedFiles
     is not available").
  3. web/model_meta.js, web/thresholds.js and web/data/model_card.json are
     generated from that model's checkpoint, so the page's class list,
     window length and STFT size can never disagree with the file it runs.

The browser runs the SAME model as the Python Signal Pipeline: the
checkpoint configs/default.yaml names as browser_model (60 classes),
exported with

    python scripts/export_onnx.py --ckpt <browser_model>

which writes the .onnx next to it. The old 8-class
files (best_model.onnx + ensemble_0..4.onnx, 26 Sept 2026) are in git history.

Usage:
    python web/build.py
"""
import base64
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
MODELS_OUT = WEB / "models"
# The model the browser runs: the same checkpoint src/pipeline/classify.py
# loads. Its ONNX export sits next to it (scripts/export_onnx.py --ckpt).
def _browser_ckpt():
    from src.config import CFG
    return REPO / CFG.get("browser_model", "results/classifier/physics_aug_best.pt")


BROWSER_CKPT = _browser_ckpt()
BROWSER_ONNX = BROWSER_CKPT.with_suffix(".onnx")
WEB_MODEL = "best_model"          # served as /models/best_model.onnx (serve_local swaps in retrained ones)
# Files from the 8-class era that must not linger next to the 60-class model.
STALE_MODELS = [f"ensemble_{i}" for i in range(5)]
# The browser model's decision line: configs/default.yaml web_threshold (one
# value for every class; see the comment there), used when the config does
# not set it.
WEB_THRESHOLD = 0.3
# Operational tier of each family (configs/default.yaml class_groups), for the
# page's colours, ribbon and alerts. Classes named here override their family.
FAMILY_TIER = {"JAMMING": "Hostile", "RADAR": "Military", "SPREAD": "Military",
               "PULSE_COMMS": "Military"}
CLASS_TIER = {"NOISE_FLOOR": "Empty", "LORA_CSS": "Civilian", "AFSK": "Civilian"}


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
    out = out.replace("__BROWSER_MODEL__", BROWSER_CKPT.name)

    # Supabase credentials. Only the DEPLOYED site gets a key: analyses from
    # the live app go to the shared table, while a development run stays in
    # the developer's own browser. The table is then a record of what the live
    # site actually saw, not a mix of that and everyone's experiments.
    #
    # A local build therefore always leaves the key empty, which storage.js
    # reads as "no key". The deploy fills it in from SUPABASE_ANON_KEY (see
    # web/vercel-build.mjs), so nothing a developer builds and commits can
    # carry a key into this public repository.
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    url = os.environ.get("SUPABASE_URL", "https://yoirhstytgrhvfunxvlc.supabase.co")
    out = out.replace("__SUPABASE_URL__", url)
    out = out.replace("__SUPABASE_ANON_KEY__", key)
    print("  supabase   " + ("key injected from SUPABASE_ANON_KEY"
                              if key else "no SUPABASE_ANON_KEY set -- browser storage only"))
    (WEB / "index.html").write_text(out, encoding="utf-8")
    print(f"  index.html  ({len(out) / 1024:.0f} KB)")


def _browser_meta():
    """The browser model's checkpoint meta (classes, window, STFT, flags,
    provenance) and its parameter counts. None when torch or the
    checkpoint is missing on this machine."""
    try:
        import torch
    except ImportError:
        print("  torch not installed -- model_meta.js / thresholds.js / model_card.json left as they are")
        return None
    if not BROWSER_CKPT.is_file():
        print(f"  warning: {BROWSER_CKPT.relative_to(REPO)} missing -- model_meta.js / thresholds.js / "
              "model_card.json left as they are")
        return None
    from src.models.amc_cnn import AMC_CNN
    ck = torch.load(BROWSER_CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = AMC_CNN(num_classes=len(meta["classes"]), input_len=meta["input_len"],
                    stft_n_fft=meta["stft_n_fft"], **meta.get("model_flags", {}))
    model.load_state_dict(ck["state_dict"])

    def count(m):
        return int(sum(p.numel() for p in m.parameters())) if m is not None else 0

    branches = {name: count(getattr(model, name)) for name in
                ("iq_branch", "stft_branch", "expert_norm", "spectrum_branch", "physics_norm")
                if getattr(model, name, None) is not None}
    return {"meta": meta, "parameters": count(model), "branches": branches,
            "hop": model.stft_branch.hop_length}


def build_model_meta():
    """web/model_meta.js, web/thresholds.js and web/data/model_card.json from
    the browser model's checkpoint.

    Generated rather than hand-written for the reason thresholds.js already
    was: a hand-maintained copy drifts. The page used to hardcode 8 classes
    and a 16-point STFT; the 60-class model needs a 64-point STFT and a
    different class order, and one stale constant would feed it
    out-of-distribution input without any error. Committed like index.html
    and web/models/*.onnx, so a clone that never runs this script still has
    a working page.
    """
    import datetime
    import hashlib
    import json

    from src.config import CFG, CLASS_GROUPS

    info = _browser_meta()
    if info is None:
        return
    meta = info["meta"]
    classes = list(meta["classes"])
    groups = {g: [c for c in cs if c in classes] for g, cs in CLASS_GROUPS.items()}
    groups["NOISE"] = ["NOISE_FLOOR"]
    family = {c: g for g, cs in groups.items() for c in cs}
    missing = [c for c in classes if c not in family]
    if missing:
        raise SystemExit(f"classes with no family in configs/default.yaml class_groups: {missing}")
    tier = {c: CLASS_TIER.get(c, FAMILY_TIER.get(family[c], "Civilian")) for c in classes}
    onnx_sha = (hashlib.sha256(BROWSER_ONNX.read_bytes()).hexdigest()[:12]
                if BROWSER_ONNX.is_file() else None)
    model = {
        "checkpoint": BROWSER_CKPT.relative_to(REPO).as_posix(),
        "name": BROWSER_CKPT.name,
        "onnx_sha256_12": onnx_sha,
        "data_version": meta.get("combined_version"),
        "seed": meta.get("seed"), "epoch": meta.get("epoch"),
        "val_top1": (meta.get("val") or {}).get("top1"),
        "flags": meta.get("model_flags", {}),
        "git_commit": meta.get("git_commit"),
    }

    def js(v):
        return json.dumps(v, indent=1)

    src = (
        "// GENERATED by web/build.py from " + model["checkpoint"] + " (its own meta)\n"
        "// and configs/default.yaml's class_groups. Do not edit by hand -- re-run\n"
        "// `python web/build.py` after exporting a new model (scripts/export_onnx.py --ckpt).\n"
        "//\n"
        "// The browser model (models/best_model.onnx) is the same checkpoint the\n"
        "// Python Signal Pipeline runs (src/pipeline/classify.py).\n"
        f"export const MODEL = {js(model)};\n"
        f"export const CLASSES = {json.dumps(classes)};\n"
        f"export const WINDOW_LEN = {int(meta['input_len'])};\n"
        f"export const STFT_N_FFT = {int(meta['stft_n_fft'])};\n"
        f"export const STFT_HOP = {int(info['hop'])};\n"
        f"export const FS = {int(CFG['signal']['fs'])};\n"
        "// family -> its classes (configs/default.yaml class_groups, plus NOISE)\n"
        f"export const GROUPS = {js(groups)};\n"
        f"export const FAMILY_OF = {json.dumps(family)};\n"
        "// operational tier per class: web/build.py FAMILY_TIER / CLASS_TIER\n"
        f"export const TIER_OF = {json.dumps(tier)};\n"
    )
    (WEB / "model_meta.js").write_text(src, encoding="utf-8")
    print(f"  model_meta.js  {len(classes)} classes, window {meta['input_len']}, "
          f"STFT {meta['stft_n_fft']}/{info['hop']}, from {model['name']}")

    per_class = CFG.get("web_thresholds_per_class") or {}
    line = float(CFG.get("web_threshold", WEB_THRESHOLD))
    values = {c: float(per_class.get(c, line)) for c in classes}
    body = ",\n".join(f"  {json.dumps(c)}: {v:g}" for c, v in values.items())
    (WEB / "thresholds.js").write_text(
        "// GENERATED by web/build.py. Do not edit by hand.\n"
        "//\n"
        "// The decision line per class for the browser model. The 60-class model\n"
        "// has no calibrated per-class thresholds yet: every class is reported\n"
        f"// once its sigmoid score passes {line:g} (configs/default.yaml web_threshold),\n"
        "// unless configs/default.yaml lists one under web_thresholds_per_class.\n"
        "export const THRESHOLDS = {\n" + body + ",\n};\n", encoding="utf-8")
    print(f"  thresholds.js  {len(per_class)} set per class, the rest {line:g}")

    from src.measure import (C42_BOUNDARY, C42_MIN_SNR_DB, C42_POOLED_ACCURACY,
                              MIN_WINDOWS_FOR_C42_DECISION)
    card = {
        "architecture": "AMC_CNN",
        "member_architecture": "AMC_CNN",
        "name": model["name"],
        "data_version": model["data_version"], "seed": model["seed"], "epoch": model["epoch"],
        "val_top1": model["val_top1"],
        "flags": model["flags"],
        "parameters": info["parameters"],
        "branches": info["branches"],
        "classes": classes,
        "families": groups,
        "tiers": tier,
        "window_len": int(meta["input_len"]),
        "fs": CFG["signal"]["fs"],
        "thresholds": values,
        "benchmark_recall": CFG["benchmark_recall"],
        "judged_classes": CFG["judged_classes"],
        "c42": {
            "boundary": C42_BOUNDARY,
            "min_windows": MIN_WINDOWS_FOR_C42_DECISION,
            "min_snr_db": C42_MIN_SNR_DB,
            "pooled_accuracy": {str(k): v for k, v in C42_POOLED_ACCURACY.items()},
        },
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "data" / "model_card.json").write_text(json.dumps(card, indent=1))
    print(f"  data/model_card.json  ({card['parameters']:,} params)")


def build_models():
    # Imported here, not at module scope: editing the page (the common case)
    # must not require the onnx package. Without this, a machine that only
    # writes HTML/JS cannot regenerate index.html at all -- it fails on the
    # import before reaching build_html().
    try:
        import onnx
    except ImportError:
        print("  onnx not installed -- skipping model copy (pip install onnx to refresh "
               "web/models/). index.html was still rebuilt.")
        return

    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    if not BROWSER_ONNX.is_file():
        print(f"  warning: {BROWSER_ONNX.relative_to(REPO)} missing -- run "
              f"python scripts/export_onnx.py --ckpt {BROWSER_CKPT.relative_to(REPO).as_posix()} first")
        return
    model = onnx.load(str(BROWSER_ONNX))          # pulls in a sidecar .data if there is one
    n_out = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    dst = MODELS_OUT / f"{WEB_MODEL}.onnx"
    onnx.save_model(model, str(dst), save_as_external_data=False)
    print(f"  models/{WEB_MODEL}.onnx  ({dst.stat().st_size / 1024:.0f} KB, {n_out} classes, "
          f"from {BROWSER_ONNX.relative_to(REPO).as_posix()})")
    for name in STALE_MODELS:
        old = MODELS_OUT / f"{name}.onnx"
        if old.exists():
            old.unlink()
            print(f"  models/{name}.onnx  removed (8-class era; in git history)")


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

    # A model/checkpoint mismatch must not stop the page being rebuilt. This
    # step needs results/*.pt to match the ARCHITECTURE the config currently
    # selects, and the config carries experiment flags (stft_keep_rows and
    # friends) that are turned on well before the matching checkpoints exist.
    # Editing the HTML then failed with a wall of missing state_dict keys,
    # after index.html had already been written -- the build looked broken
    # when only its last step was. The existing web/data/performance.json is
    # left in place and the page keeps showing the numbers it was built with.
    try:
        model = load_model("auto")
        breakdown = single_vs_multi(model, X[test], y[test], snr[test],
                                     classes=list(CLASSES))
    except Exception as e:
        first = str(e).strip().splitlines()[0]
        print(f"  warning: performance data NOT refreshed -- {first}")
        print("           results/*.pt do not match the architecture in "
               "configs/default.yaml (check the model: flags), so the Performance "
               "page keeps its previous numbers.")
        return
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

    # scorecard.json records nothing about WHICH model wrote it -- src.evaluate
    # writes the same filename whether run plain (single checkpoint) or with
    # --ensemble. Rather than guess, compare it against the ensemble's own
    # judged-class recalls: if they agree to the last digit it was the same
    # model, because two different models do not produce identical recalls on
    # 3060 windows by chance.
    # src.evaluate records this directly now (scorecard.json's `provenance`
    # block), so prefer it and stop guessing.
    prov = (scorecard or {}).get("provenance") or {}

    same_as_ensemble = False
    if scorecard and ensemble_scorecard and ensemble_scorecard.get("ensemble"):
        per_class = scorecard.get("per_class", {})
        same_as_ensemble = all(
            cls in per_class
            and abs(per_class[cls]["recall"] - recall) < 1e-9
            for cls, recall in ensemble_scorecard["ensemble"].items())

    if "ensemble" in prov:
        n = len(prov.get("checkpoints") or []) or "?"
        scorecard_source = (
            f"evals/scorecard.json — provenance says {n}-model ensemble "
            f"({', '.join(prov['checkpoints'])})"
            if prov["ensemble"] else
            f"evals/scorecard.json — provenance says single checkpoint "
            f"({', '.join(prov.get('checkpoints') or ['best_model.pt'])})")
    elif same_as_ensemble:
        scorecard_source = (
            "evals/scorecard.json — recalls match evals/ensemble_scorecard.json, "
            f"so this is the {ensemble_scorecard['n_models']}-model ensemble "
            "(src.evaluate --ensemble)")
    elif ensemble_scorecard:
        scorecard_source = (
            "evals/scorecard.json — differs from the ensemble scorecard, so this is "
            "src.evaluate's default: the single checkpoint (best_model.pt)")
    else:
        # Previously this branch asserted "single checkpoint", which is a guess
        # and was wrong for a scorecard written by src.evaluate --ensemble with
        # no ensemble_scorecard.json beside it. Say what is known instead.
        scorecard_source = (
            "evals/scorecard.json — no ensemble_scorecard.json and no provenance "
            "block, so which checkpoint wrote it is not recorded. Re-run "
            "python -m src.evaluate to stamp it.")
    # The figures src/evaluate.py writes. Copied rather than redrawn: they are
    # the evaluation's own output, and a chart the browser drew from the same
    # numbers would be a second rendering that could disagree with the one the
    # team circulates. src/ui/pages/performance.py embeds them as-is for the
    # same reason, and says restyling them is not that package's call.
    figures = {}
    for name in ("confusion_matrix.png", "accuracy_vs_snr.png"):
        src = REPO / "evals" / name
        if not src.is_file():
            print(f"  warning: evals/{name} missing — run python -m src.evaluate")
            continue
        shutil.copyfile(src, data_out / name)
        figures[name] = {
            "file": name,
            # Lets the page flag a figure that predates the thresholds it is
            # displayed under: evaluate.py writes the PNGs and scorecard.json
            # in the same call, so a figure older than the config was produced
            # under different thresholds.
            "mtime": datetime.datetime.fromtimestamp(
                src.stat().st_mtime).isoformat(timespec="seconds"),
        }
        print(f"  data/{name}  ({src.stat().st_size / 1024:.0f} KB)")

    (data_out / "performance.json").write_text(json.dumps({
        # Two DIFFERENT models are represented on this page and conflating
        # them would be a provenance error of exactly the kind
        # src/ui/pages/performance.py exists to prevent:
        #   - the scorecard table comes from src.evaluate, which defaults to
        #     the SINGLE checkpoint (best_model.pt) unless run --ensemble,
        #     and records nothing in the file about which it was;
        #   - the breakdown chart is computed above with load_model("auto"),
        #     i.e. the ensemble whenever all five members are present.
        "scorecard_source": scorecard_source,
        "breakdown_model": model_label("auto"),
        "figures": figures,
        # The config's own mtime, so the page can warn if a figure predates
        # the thresholds it is displayed under.
        "config_mtime": datetime.datetime.fromtimestamp(
            (REPO / "configs" / "default.yaml").stat().st_mtime).isoformat(timespec="seconds"),
        # Suppressed when the table above is already the same run -- showing
        # one model's numbers twice under two headings invites the reader to
        # treat them as independent corroboration.
        "ensemble_scorecard": None if same_as_ensemble else ensemble_scorecard,
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
        # Dense-QAM order resolution, shown alongside the per-window scorecard
        # because the per-window 16QAM/64QAM split is a coin flip the model
        # cannot win (src/evaluate.py:dense_qam_recall documents the measured
        # 51.4% single-window / 49.7% at 64 windows for the model's own
        # probabilities). The resolver is the C42 estimator in src/measure.py,
        # which pools ACROSS windows -- so this is an event-level figure, not a
        # per-window one, and is labelled as such on the page. The SNR gate
        # travels with it: below min_snr_db constellation_order refuses to
        # decide rather than guessing, so quoting the pooled accuracy without
        # that caveat would overstate what the system delivers.
        "dense_qam": {
            "c42_boundary": C42_BOUNDARY,
            "min_windows": MIN_WINDOWS_FOR_C42_DECISION,
            "min_snr_db": C42_MIN_SNR_DB,
            "pooled_accuracy": {str(k): v for k, v in C42_POOLED_ACCURACY.items()},
        },
        "scorecard_mtime": (datetime.datetime.fromtimestamp(
            scorecard_path.stat().st_mtime).isoformat(timespec="seconds")
            if scorecard_path.is_file() else None),
    }))
    print(f"  data/performance.json  ({len(test)} test windows, "
          f"{breakdown.n_windows} single/multi)")

    # The figures src/evaluate.py writes. Copied rather than redrawn: they are
    # the evaluation's own output, and a chart the browser drew from the same
    # numbers would be a second rendering that could disagree with the one the
    # team circulates. src/ui/pages/performance.py embeds them as-is for the
    # same reason and says restyling them is not that package's call.

    # The model card is written by build_model_meta(), from the browser model.


if __name__ == "__main__":
    import argparse

    # Editing the page and refreshing its DATA are different jobs, and running
    # the second by accident is expensive: build_data() rewrites
    # web/data/performance.json and copies evals/*.png, so on a checkout whose
    # evals/ is older than the deployed numbers it silently REPLACES good
    # figures with stale ones. That happened during a page edit -- the
    # Performance page went back to a 28 Aug scorecard from a smaller dataset,
    # NOISE_FLOOR recall included, and the diff is a wall of JSON and PNG bytes
    # that nobody reads before committing.
    ap = argparse.ArgumentParser(
        description="Build web/ from the template and the evaluation artifacts.")
    ap.add_argument("--page-only", action="store_true",
                    help="rebuild index.html, thresholds and models but leave web/data/ "
                         "alone -- use this for any change to the page itself")
    args = ap.parse_args()

    print("building web/")
    build_html()
    # Before the data steps: the model, its class list and thresholds must
    # refresh even on a machine without the dataset -- the page decides with
    # these whether or not build_data() can run.
    build_models()
    build_model_meta()
    # The page is always rebuilt; the data steps need things a machine editing
    # the HTML may simply not have -- the 330 MB dataset (untracked), or
    # checkpoints matching the config's current experiment flags. Missing
    # either one is reported as a warning against the files it would have
    # refreshed, not as a traceback after index.html has already been written.
    try:
        if args.page_only:
            print("  --page-only: web/data/ left untouched")
        else:
            build_data()
    except FileNotFoundError as e:
        print(f"  warning: page data NOT refreshed -- {e.filename or e}")
        print("           run scripts/build_dataset.py, or ignore this when only "
               "editing the page: web/data/ keeps its previous contents.")
    except ImportError as e:
        # numpy/torch/scipy are not installed everywhere the page is edited.
        print(f"  warning: page data NOT refreshed -- {e}")
        print("           pip install -r requirements.txt to refresh web/data/.")
    print("done")
