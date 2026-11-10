"""
Write web/data/current_model.json for the dashboard's Performance page: the
CURRENT 60-class model's test results per family and per class, from the
files a Colab run leaves in its results folder (e.g.
FOI26/results/2026-11-07_physics_seed4/): scorecard.json and compare.json.

    python scripts/export_current_model.py --run-dir path/to/2026-11-07_physics_seed4 \
        --name physics_aug_best.pt --vote-high 0.794

Numbers are copied, never recomputed: per class = "of the test windows of this
class, the share the model named exactly right" (top1_standalone); per family =
the same over all windows of that family (compare.json); overall top-1 and the
top-1 on clear signals (>= 10 dB) from compare.json.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_GROUPS, REPO_ROOT  # noqa: E402

FAMILY_PLAIN = {
    "JAMMING": "Jammers (signals meant to block radios)", "RADAR": "Radars", "SPREAD": "Spread spectrum (hopping, LoRa)",
    "FSK": "FSK data (tone-switching data)", "ASK": "ASK data (on/off data)", "ANALOG": "Analog voice radio (AM, FM)",
    "PULSE_COMMS": "Pulse links (IFF, PPM)", "PSK": "PSK data (phase-switching data)", "MULTICARRIER": "Multicarrier (Wi-Fi, 4G, 5G-like)",
    "APSK": "APSK (satellite data)", "NOISE": "Empty band (noise only)", "QAM": "QAM (dense high-speed data)",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="folder with scorecard.json and compare.json")
    p.add_argument("--name", default="physics_aug_best.pt", help="model file name shown on the page")
    p.add_argument("--run-label", default=None, help="run name shown on the page (default: the folder name)")
    p.add_argument("--vote-high", type=float, default=None,
                   help="share right when voting over 16 windows at >= 10 dB (from the run notes), optional")
    p.add_argument("--compare-key", default=None,
                   help="this run's entry in compare.json (default: the one named '... this run ...')")
    p.add_argument("--out", default=str(REPO_ROOT / "web" / "data" / "current_model.json"))
    a = p.parse_args()
    run = Path(a.run_dir)
    sc = json.loads((run / "scorecard.json").read_text(encoding="utf-8"))
    cmp_ = json.loads((run / "compare.json").read_text(encoding="utf-8"))
    mine = cmp_[a.compare_key] if a.compare_key else next(v for k, v in cmp_.items() if "this run" in k)
    per_class = sc["test"]["per_class"]
    groups = dict(CLASS_GROUPS)
    groups.setdefault("NOISE", ["NOISE_FLOOR"])
    families = []
    for fam, acc in sorted(mine["family"].items(), key=lambda kv: -kv[1]):
        classes = [{"name": c, "right": round(per_class[c]["top1_standalone"], 4)}
                   for c in groups.get(fam, []) if c in per_class]
        families.append({"family": fam, "plain": FAMILY_PLAIN.get(fam, fam), "right": round(acc, 4),
                         "classes": sorted(classes, key=lambda c: -c["right"])})
    meta = sc.get("meta", {})
    doc = {"model": a.name, "run": a.run_label or run.name, "data_version": meta.get("combined_version"),
           "seed": meta.get("seed"), "epoch": meta.get("epoch"), "git_commit": meta.get("git_commit"),
           "n_classes": len(per_class), "top1": round(mine["top1"], 4), "top1_clear": round(mine["top1_high"], 4),
           "vote_clear": a.vote_high, "families": families,
           "exported": datetime.datetime.now().isoformat(timespec="seconds")}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {a.out}: {len(families)} families, top-1 {doc['top1']:.1%}, clear {doc['top1_clear']:.1%}")


if __name__ == "__main__":
    main()
