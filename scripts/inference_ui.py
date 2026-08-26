"""Local inference UI: OMNI RF situational-awareness console.

Runs locally, not as a claude.ai Artifact -- an Artifact is sandboxed JS with
no Python and no way to load a .pt checkpoint or call the real model.

Usage:
    python scripts/inference_ui.py
Then open the printed http://127.0.0.1:7860 link.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.app import launch  # noqa: E402
import os

if __name__ == "__main__":
    launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))