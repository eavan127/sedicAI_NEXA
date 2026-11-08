"""
Morse (CW) part of the military radio evaluation set: REAL decoded traffic from
the Indonesia CW archive, RE-KEYED in simulation as IQ. Demo and validation
only, never training data.

The archive recordings themselves (8 kHz audio, whole days) are not needed:
scripts/comms_intel.py already found every transmission in them and wrote
results/comms_intel/<recording>.json (start, speed, tone, SNR, decoded text).
This script takes the readable, strong transmissions (unreadable < 5%,
SNR >= 15 dB), cuts an excerpt of up to --max-s seconds at the transmission's
own speed, and keys it on a CW carrier the way a transmitter does: dots and
dashes with soft edges and a little human timing jitter, at a random offset
within +-80 kHz and 12-30 dB SNR (3 kHz band), as 2-channel int16 IQ at
250 kS/s (scripts/run_pipeline.py --iq reads it). Label for the exhibition:
"real Morse traffic, re-transmitted in simulation", never "live intercept".

    python scripts/make_morse_dataset.py
    python scripts/make_morse_dataset.py --out D:/somewhere/morse --n 40 --max-s 12

Output (default ~/Downloads/military_radio_IQ/morse, next to english/ malay/ chinese/):
  iq/<id>_CW.wav     the RF capture
  audio/<id>.wav     the 700 Hz beep a receiver in CW mode plays (16 kHz)
  manifest.csv       source recording and time, speed, tone, SNR, text, offset, timing
"""
import argparse
import csv
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402
from src.intel.morse import TO_CODE, noise_like  # noqa: E402

MIN_S = 5.0                     # shorter excerpts are decoder noise, not messages
FA = 50e3
FS_OUT = 250e3
AUDIO_FS = 16000
BEEP_HZ = 700.0
SRC = REPO_ROOT / "results" / "comms_intel"
OUT = Path.home() / "Downloads" / "military_radio_IQ" / "morse"
TOKEN = re.compile(r"<[A-Z]{2}>|.")
README_NOTE = """
morse/    {n} items: real decoded CW traffic from the Indonesia CW archive (results/comms_intel),
          re-keyed in simulation at each transmission's own speed. iq/<id>_CW.wav, audio/<id>.wav
          (700 Hz beep), manifest.csv with the source recording, time, WPM and text.
          Made by scripts/make_morse_dataset.py. Say "real Morse traffic, re-transmitted in simulation".
"""


def clean_words(text):
    """Decoder output -> keyable words: unreadable marks (*) removed, and the
    long E/I/T/S/H runs the decoder makes from key-down noise dropped."""
    words = []
    for w in text.replace("*", "").split():
        if len(w) >= 6 and set(w) <= set("EITSH"):
            continue
        toks = [t for t in TOKEN.findall(w) if t in TO_CODE]
        if toks:
            words.append(toks)
    return words


def units(words):
    """Words (lists of characters) -> [(on, length_in_dots)] runs."""
    out = []
    for wi, w in enumerate(words):
        if wi:
            out.append((0, 7))
        for ci, ch in enumerate(w):
            if ci:
                out.append((0, 3))
            for ei, el in enumerate(TO_CODE[ch]):
                if ei:
                    out.append((0, 1))
                out.append((1, 1 if el == "." else 3))
    return out


def excerpt(words, wpm, max_s):
    """The first words that fit into max_s at this speed."""
    dot = 1.2 / wpm
    keep, total = [], 0.0
    for w in words:
        n = sum(L for _, L in units([w])) + (7 if keep else 0)
        if keep and (total + n) * dot > max_s:
            break
        keep.append(w)
        total += n
    return keep


def key_gate(words, wpm, fs, rng, jitter=0.06):
    """On/off keying at fs: element lengths with human jitter, 5 ms soft edges."""
    dot = 1.2 / wpm
    parts = []
    for on, L in units(words):
        n = max(1, int(round(L * dot * fs * (1 + rng.normal(0, jitter)))))
        parts.append(np.full(n, float(on)))
    g = np.concatenate(parts)
    ramp = int(0.005 * fs)
    return np.convolve(g, np.ones(ramp) / ramp, mode="same")


def write_wav(path, x, fs, ch=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def write_iq(path, iq):
    scale = 0.9 / np.abs(np.concatenate([iq.real, iq.imag])).max()
    inter = np.empty(2 * len(iq))
    inter[0::2], inter[1::2] = iq.real * scale, iq.imag * scale
    write_wav(path, inter, FS_OUT, ch=2)


def candidates(src, max_s, per_recording):
    rows, seen = [], set()
    for f in sorted(src.glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        if "segments" not in j:
            continue
        n = 0
        for s in sorted(j["segments"], key=lambda s: -s.get("snr_db", 0)):
            if s.get("unreadable", 1) >= 0.05 or s.get("snr_db", 0) < 15 or not s.get("wpm"):
                continue
            words = excerpt(clean_words(s.get("text", "")), s["wpm"], max_s)
            text = " ".join("".join(w) for w in words)
            secs = sum(L for _, L in units(words)) * 1.2 / s["wpm"] if words else 0
            # Key-down noise decodes as short E/I/T bursts at odd speeds: not traffic.
            if len(words) < 3 or secs < MIN_S or text in seen or noise_like(text, s["wpm"]):
                continue
            seen.add(text)
            rows.append({"source_recording": j.get("file", f.stem + ".wav"), "recorded_at": j.get("recorded_at"),
                         "source_start_s": round(s["start_s"], 1), "source_end_s": round(s["end_s"], 1),
                         "wpm": round(s["wpm"], 1), "tone_hz_original": round(s.get("tone_hz", 0), 1),
                         "snr_db_original": round(s.get("snr_db", 0), 1), "text": text, "_words": words})
            n += 1
            if n >= per_recording:
                break
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC), help="comms_intel results folder")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--n", type=int, default=36, help="clips at most")
    p.add_argument("--per-recording", type=int, default=4)
    p.add_argument("--max-s", type=float, default=12.0, help="excerpt length at most (s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for sub in ("iq", "audio"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = candidates(Path(args.src), args.max_s, args.per_recording)[:args.n]
    if not rows:
        sys.exit(f"no usable transmissions in {args.src}")
    for i, r in enumerate(rows, 1):
        words = r.pop("_words")
        rid = f"cw_{i:03d}"
        gate_a = key_gate(words, r["wpm"], AUDIO_FS, np.random.default_rng(args.seed * 1000 + i))
        lead_a = np.zeros(int(0.3 * AUDIO_FS))
        beep = np.concatenate([lead_a, gate_a, lead_a])
        beep = 0.7 * beep * np.sin(2 * np.pi * BEEP_HZ * np.arange(len(beep)) / AUDIO_FS)
        write_wav(out / "audio" / f"{rid}.wav", beep, AUDIO_FS)
        gate = key_gate(words, r["wpm"], FA, np.random.default_rng(args.seed * 1000 + i)).astype(complex)
        offset = float(np.round(rng.uniform(-80e3, 80e3), -2))
        snr = float(np.round(rng.uniform(12, 30), 1))
        on = gate.real > 0.5
        bb = gate / np.sqrt(np.mean(np.abs(gate[on]) ** 2)) * np.sqrt(10 ** (snr / 10) * 3000 / FS_OUT)
        w = resample_poly(bb, int(FS_OUT / FA), 1)
        lead = int(0.3 * FS_OUT)
        n = lead + len(w) + lead
        iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq[lead:lead + len(w)] += w * np.exp(2j * np.pi * offset * (lead + np.arange(len(w))) / FS_OUT)
        name = f"{rid}_CW.wav"
        write_iq(out / "iq" / name, iq)
        r.update({"id": rid, "iq_file": f"iq/{name}", "modulation": "CW", "offset_khz": offset / 1e3,
                  "snr_db_3khz": snr, "fs": FS_OUT, "tx_start_s": 0.3, "tx_end_s": round((lead + len(w)) / FS_OUT, 3),
                  "duration_s": round(n / FS_OUT, 3),
                  "note": "real decoded CW traffic, re-keyed in simulation"})
        print(f"{rid}  {r['wpm']:4.1f} wpm  {offset / 1e3:+6.1f} kHz {snr:4.1f} dB  {r['duration_s']:5.1f} s  {r['text'][:60]}")
    keys = ["id", "iq_file", "modulation", "text", "wpm", "offset_khz", "snr_db_3khz", "fs", "tx_start_s", "tx_end_s",
            "duration_s", "source_recording", "recorded_at", "source_start_s", "source_end_s", "tone_hz_original",
            "snr_db_original", "note"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    readme = out.parent / "README.txt"
    if readme.exists() and "morse/" not in readme.read_text(encoding="utf-8"):
        readme.write_text(readme.read_text(encoding="utf-8").rstrip() + "\n" + README_NOTE.format(n=len(rows)),
                          encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{len(rows)} Morse clips from {len({r['source_recording'] for r in rows})} recordings, {size:.0f} MB in {out}")


if __name__ == "__main__":
    main()
