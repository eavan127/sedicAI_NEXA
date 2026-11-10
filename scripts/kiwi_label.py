"""
Pre-label KiwiSDR IQ recordings for the real-world HF test set, from evidence
that does NOT come from NEXA's classifier (testing NEXA on its own answers
would be circular):

  * the signal's measured shape: occupied width, on/off keying, two tones a
    fixed shift apart, a thin constant line, a voice-wide band that rises and
    falls like speech, or nothing above the noise;
  * the known use of the frequency (amateur band plan, listed stations).

Every label is a SUGGESTION with a confidence and the evidence behind it. A
person confirms each one by listening (docs/KIWI_LABELLING.md) and fills in
human_label; only confirmed clips go into the test set.

    python scripts/kiwi_label.py data/kiwi                 # every .wav in the folder
    python scripts/kiwi_label.py data/kiwi --sheet data/kiwi/labels.csv

Recordings: 2-channel int16 IQ WAV from kiwirecorder -m iq (12 or 20.25 kS/s),
centred on the frequency given with -f. The frequency is read from the file
name kiwirecorder writes (..._<Hz>_iq.wav) or from a name like ssb_14250.wav.
"""
import argparse
import csv
import re
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import welch

LABELS = ("noise", "cw", "rtty", "psk31", "ft8", "ssb_voice", "am", "unknown")
# Known uses of HF frequencies (kHz): (lo, hi, likely label, what). Matched on
# the frequency of the SIGNAL (centre + its offset), not of the recording.
FT8_DIALS = (1840.0, 3573.0, 7074.0, 10136.0, 14074.0, 18100.0, 21074.0, 24915.0, 28074.0)
FREQ_HINTS = [(d, d + 3.0, "ft8", f"amateur FT8 {d:g} kHz (8 slow tones)") for d in FT8_DIALS] + [
    (10100.0, 10101.6, "rtty", "DWD weather RTTY (Germany) 10100.8 kHz"),
    (4582.0, 4584.0, "rtty", "DWD weather RTTY 4583 kHz"),
    (7645.0, 7647.0, "rtty", "DWD weather RTTY 7646 kHz"),
    (14466.0, 14468.5, "rtty", "DWD weather RTTY 14467.3 kHz"),
    (7038.0, 7042.0, "psk31", "amateur PSK31 area 7040 kHz"),
    (7070.0, 7073.0, "psk31", "amateur PSK31 area 7070 kHz"),
    (14069.0, 14074.0, "psk31", "amateur PSK31 area 14070 kHz"),
    (21069.0, 21074.0, "psk31", "amateur PSK31 area 21070 kHz"),
    (7000.0, 7038.0, "cw", "amateur CW segment 7000-7035 kHz"),
    (14000.0, 14069.0, "cw", "amateur CW segment 14000-14060 kHz"),
    (21000.0, 21069.0, "cw", "amateur CW segment 21000-21070 kHz"),
    (7100.0, 7300.0, "ssb_voice", "amateur phone 40 m (LSB)"),
    (14150.0, 14350.0, "ssb_voice", "amateur phone 20 m (USB)"),
    (21200.0, 21450.0, "ssb_voice", "amateur phone 15 m (USB)"),
    (9995.0, 10005.0, "am", "time stations (WWV / BPM) 10 MHz"),
    (14995.0, 15005.0, "am", "time stations (WWV / BPM) 15 MHz"),
]
RTTY_SHIFTS = (170.0, 425.0, 450.0, 850.0)


def read_iq(path):
    with wave.open(str(path), "rb") as w:
        fs, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if ch != 2 or width != 2:
        raise ValueError(f"{path.name}: need 2-channel int16 IQ (kiwirecorder -m iq)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64), float(fs)


def freq_khz_of(path):
    m = re.search(r"_(\d{6,9})_iq", path.name)              # kiwirecorder: ..._<Hz>_iq.wav
    if m:
        return int(m.group(1)) / 1e3
    m = re.search(r"_(\d{3,5}(?:\.\d+)?)(?:khz)?(?:_|\.wav)", path.name, re.I)   # ssb_14250.wav
    return float(m.group(1)) if m else None


def hint_for(f_khz):
    for lo, hi, label, what in FREQ_HINTS:
        if f_khz is not None and lo <= f_khz <= hi:
            return label, what
    return None, None


def measure(iq, fs):
    """Signal shape, from the spectrum and the envelope only."""
    nfft = int(2 ** np.ceil(np.log2(fs / 5)))               # ~5 Hz bins
    f, p = welch(iq - iq.mean(), fs=fs, nperseg=min(nfft, len(iq) // 4), return_onesided=False)
    order = np.argsort(f)
    f, p = f[order], p[order]
    keep = np.abs(f) < 0.45 * fs                            # the receiver's own filter edges
    f, p = f[keep], p[keep]
    db = 10 * np.log10(p + 1e-20)
    floor = np.median(db)
    peak = float(db.max() - floor)
    # Occupied width: within 20 dB of the peak (and above the noise), so that
    # a strong signal's faint key clicks or splatter do not count.
    hot = db > max(floor + 6, db.max() - 20)
    width = float(hot.sum() * (f[1] - f[0])) if hot.any() else 0.0
    # Two-tone (FSK) check: the two strongest peaks at least 100 Hz apart.
    i1 = int(np.argmax(db))
    away = np.abs(f - f[i1]) > 100
    i2 = int(np.argmax(np.where(away, db, -np.inf)))
    tone_gap = float(abs(f[i2] - f[i1]))
    tone_balance = float(db[i1] - db[i2])
    # Envelope over time: keyed (Morse), syllabic (speech) or steady.
    blk = int(0.01 * fs)
    env = np.sqrt(np.convolve(np.abs(iq) ** 2, np.ones(blk) / blk, mode="valid")[::blk])
    env = env / (np.percentile(env, 95) + 1e-12)
    on = env > 0.5
    flips = float(np.mean(on[1:] != on[:-1])) if len(on) > 1 else 0.0
    spread = float(np.std(env))
    # Sidebands around a strong line (AM carrier with audio, e.g. time stations).
    around = (np.abs(f - f[i1]) > 200) & (np.abs(f - f[i1]) < 5000)
    side = float(np.percentile(db[around], 99) - floor) if around.any() else 0.0
    return {"peak_db": peak, "peak_hz": float(f[i1]), "width_hz": width, "tone_gap_hz": tone_gap,
            "tone_balance_db": tone_balance, "sideband_db": side,
            "on_share": float(on.mean()), "on_off_flips": flips, "envelope_spread": spread}


def suggest(m, hint):
    """-> (label, confidence 0..1, evidence)."""
    if m["peak_db"] < 6:
        return "noise", 0.9, f"nothing stands {m['peak_db']:.0f} dB above the noise"
    shift = min(RTTY_SHIFTS, key=lambda s: abs(s - m["tone_gap_hz"]))
    if abs(shift - m["tone_gap_hz"]) < 25 and m["tone_balance_db"] < 6 and m["width_hz"] < 1500:
        return "rtty", 0.8, f"two tones {m['tone_gap_hz']:.0f} Hz apart (standard shift {shift:.0f} Hz)"
    if m["width_hz"] < 150 and m["on_off_flips"] > 0.05 and 0.15 < m["on_share"] < 0.85:
        return "cw", 0.8, f"narrow ({m['width_hz']:.0f} Hz) and switching on/off ({100 * m['on_share']:.0f}% on)"
    if m["width_hz"] < 150 and m["peak_db"] >= 20 and m["sideband_db"] >= 6 and hint == "am":
        return "am", 0.7, f"strong carrier ({m['peak_db']:.0f} dB) with sidebands {m['sideband_db']:.0f} dB above the noise"
    if m["width_hz"] < 150 and m["envelope_spread"] < 0.35:
        lab = hint if hint in ("psk31", "ft8") else "unknown"
        return lab, 0.6 if lab != "unknown" else 0.3, f"one thin steady line ({m['width_hz']:.0f} Hz wide)"
    if 800 < m["width_hz"] < 3500 and m["envelope_spread"] > 0.25:
        return "ssb_voice", 0.7, f"voice-wide ({m['width_hz']:.0f} Hz) and rising/falling like speech"
    if m["width_hz"] >= 3500 and m["envelope_spread"] > 0.2:
        return "am", 0.5, f"wide ({m['width_hz']:.0f} Hz), level changes: AM broadcast or time station?"
    return "unknown", 0.2, f"{m['width_hz']:.0f} Hz wide, {m['peak_db']:.0f} dB above the noise"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="folder with KiwiSDR IQ .wav recordings")
    p.add_argument("--sheet", default=None, help="labels CSV to write (default <folder>/labels.csv)")
    a = p.parse_args()
    folder = Path(a.folder)
    sheet = Path(a.sheet) if a.sheet else folder / "labels.csv"
    old = {}
    if sheet.exists():                                      # keep what a person already confirmed
        for r in csv.DictReader(open(sheet, encoding="utf-8-sig")):
            old[r["file"]] = r
    rows = []
    for wav in sorted(folder.glob("*.wav")):
        try:
            iq, fs = read_iq(wav)
        except Exception as e:                              # noqa: BLE001 -- note it and go on
            print(f"{wav.name}: skipped ({e})")
            continue
        f_khz = freq_khz_of(wav)
        m = measure(iq, fs)
        sig_khz = f_khz + m["peak_hz"] / 1e3 if f_khz is not None and m["peak_db"] >= 6 else f_khz
        hint, what = hint_for(sig_khz)
        label, conf, why = suggest(m, hint)
        if hint and hint != label and conf < 0.8:
            why += f"; frequency suggests {hint} ({what})"
        elif hint == label:
            conf = min(0.95, conf + 0.1)
            why += f"; matches the frequency ({what})"
        prev = old.get(wav.name, {})
        rows.append({"file": wav.name, "freq_khz": f_khz if f_khz is not None else "",
                     "signal_khz": round(sig_khz, 3) if sig_khz is not None else "", "seconds": round(len(iq) / fs, 1),
                     "suggested": label, "confidence": round(conf, 2), "evidence": why,
                     "human_label": prev.get("human_label", ""), "notes": prev.get("notes", "")})
        print(f"{wav.name:<48} {label:<10} {conf:.2f}  {why}")
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    todo = sum(not r["human_label"] for r in rows)
    print(f"{len(rows)} clips -> {sheet} ({todo} still need a human_label)")


if __name__ == "__main__":
    main()
