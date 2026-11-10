"""
Record a list of HF targets from a public KiwiSDR as IQ, one 30 s clip each,
named <guess>_<kHz>_<UTC time>.wav, then pre-label them (scripts/kiwi_label.py).
Uses kiwirecorder from the official KiwiSDR client
(git clone https://github.com/jks-prv/kiwiclient D:/tools/kiwiclient).

    python scripts/kiwi_collect.py                       # the default target list, Kuala Lumpur receiver
    python scripts/kiwi_collect.py --only rtty psk31     # some classes only
    python scripts/kiwi_collect.py --freq 7074.0 --guess unknown

Real-world TEST data only: never training (synthetic-only training rule).
Be a good guest on a shared receiver: short clips, one at a time.
"""
import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402

KIWI = Path("D:/tools/kiwiclient/kiwirecorder.py")
HOST, PORT = "21439.proxy2.kiwisdr.com", 8073          # 9M2AMH, Kuala Lumpur
# (guess, kHz the IQ is centred on). For SSB / PSK31 / RTTY the signal sits
# within the ~12 kHz IQ window around it, so the exact spot need not be known.
TARGETS = [
    ("noise", 13400.0), ("noise", 16500.0), ("noise", 19500.0), ("noise", 23000.0),
    ("cw", 7020.0), ("cw", 14030.0), ("cw", 21030.0), ("cw", 10110.0),
    ("rtty", 10100.8), ("rtty", 7646.0), ("rtty", 14467.3), ("rtty", 4583.0),
    ("psk31", 14070.5), ("psk31", 7040.5), ("psk31", 21070.5), ("psk31", 7070.5),
    ("ssb_voice", 7150.0), ("ssb_voice", 7180.0), ("ssb_voice", 14200.0), ("ssb_voice", 14250.0),
    ("ssb_voice", 14300.0), ("ssb_voice", 21300.0),
    ("am", 10000.0), ("am", 15000.0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--only", nargs="+", default=None, help="only these guesses (noise cw rtty psk31 ssb_voice am)")
    p.add_argument("--freq", type=float, default=None, help="record one frequency (kHz) instead of the list")
    p.add_argument("--guess", default="unknown", help="with --freq: name prefix")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "kiwi"))
    p.add_argument("--kiwirecorder", default=str(KIWI))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(a.kiwirecorder).exists():
        sys.exit(f"kiwirecorder not found at {a.kiwirecorder}: git clone https://github.com/jks-prv/kiwiclient D:/tools/kiwiclient")
    targets = [(a.guess, a.freq)] if a.freq else [t for t in TARGETS if not a.only or t[0] in a.only]
    for guess, khz in targets:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{guess}_{khz:g}_{stamp}"
        cmd = [sys.executable, a.kiwirecorder, "-s", a.host, "-p", str(a.port), "-f", str(khz), "-m", "iq",
               "--tlimit", str(a.seconds), "-d", str(out), "--filename", name]
        print(f"recording {name} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not any(out.glob(f"{name}*.wav")):
            print(f"  failed: {(r.stderr or r.stdout).strip().splitlines()[-1:] or ['no file']}", flush=True)
            if "too many" in (r.stderr + r.stdout).lower():
                print("  the receiver is full; try later or another receiver (--host / --port)")
                break
        time.sleep(2)                                   # be kind to the shared receiver
    print("pre-labelling ...", flush=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "kiwi_label.py"), str(out)])


if __name__ == "__main__":
    main()
