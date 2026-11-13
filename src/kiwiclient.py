"""
Where kiwirecorder.py (the official KiwiSDR client, github.com/jks-prv/kiwiclient)
is on this machine: NEXA_KIWIRECORDER if set, else the first of a few usual
folders that has it, so the Live KiwiSDR panel works without any setup step.
"""
import os
from pathlib import Path

from src.config import REPO_ROOT

FOLDERS = [REPO_ROOT / "tools" / "kiwiclient", Path.home() / "tools" / "kiwiclient",
           Path("D:/tools/kiwiclient"), Path("C:/tools/kiwiclient")]
INSTALL = r"git clone https://github.com/jks-prv/kiwiclient %USERPROFILE%\tools\kiwiclient"


def find_kiwirecorder():
    """-> Path of kiwirecorder.py (it may not exist: then where it should go)."""
    env = os.environ.get("NEXA_KIWIRECORDER")
    if env:
        return Path(env)
    for folder in FOLDERS:
        if (folder / "kiwirecorder.py").exists():
            return folder / "kiwirecorder.py"
    return Path.home() / "tools" / "kiwiclient" / "kiwirecorder.py"


def not_found(path):
    """The message for a missing kiwirecorder: where NEXA looked and how to get it."""
    looked = ", ".join(str(f) for f in FOLDERS)
    return (f"kiwirecorder not found at {path} (also looked in {looked}). "
            f"Install it with: {INSTALL}  (or set NEXA_KIWIRECORDER to its kiwirecorder.py)")
