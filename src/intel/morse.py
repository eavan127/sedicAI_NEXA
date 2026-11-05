"""Morse (CW) decoding from receiver audio.

How it works (piano version: find the one key being played, then write
down how long each press and each pause lasts):

  1. Every 15 s of audio, find the beep's pitch: the strongest narrow peak
     between 250 and 2000 Hz.
  2. Shift that pitch to 0 Hz and average over 5 ms blocks: a 200 values/s
     "loudness of the beep" track that ignores the rest of the audio band.
  3. Key down / key up with two thresholds (hysteresis) between the noise
     level and the beep level, so noise does not flicker it on and off.
     A 15 s stretch whose beep is less than 8 dB above the noise, or whose
     strongest pitch holds under 6% of the audio power (plain noise: ~3%;
     Morse at -5 dB: ~18%), counts as silent.
  4. Split into transmissions at pauses over 2 s. In each, the dot length
     is the shorter of two groups of key-down times (operators send at
     different speeds, so it is measured per transmission).
  5. Press < 2 dots = dot, else dash. Pause < 2 dots = same letter,
     < 5 dots = next letter, longer = next word.

Speed is reported in words per minute (WPM = 1.2 / dot seconds).
Unreadable letters come out as '*'; a key held down over 1.5 s as '[carrier]'.
Transmissions that look like noise read as Morse (speed over 45 WPM, or
mostly the one-element letters E / T) are dropped, counted in `dropped`.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .audio import WavInfo, iter_chunks

ENV_RATE = 200                  # envelope values per second (5 ms)
TONE_BAND = (250.0, 2000.0)
MIN_SNR_DB = 8.0
SEGMENT_GAP_S = 2.0
GLITCH = 3                      # runs shorter than 3 envelope steps (15 ms) are noise
CARRIER_S = 1.5
CHUNK_S = 15.0                  # tone, threshold and SNR are re-measured this often
MAX_WPM = 45.0
MIN_PURITY = 0.06

CODE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G",
    "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N",
    "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T", "..-": "U",
    "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5",
    "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", "-..-.": "/", "-....-": "-", "-.--.-": ")",
    ".----.": "'", "---...": ":", "-.-.-.": ";", ".-..-.": '"', ".--.-.": "@",
    "-...-": "=",            # BT, also the break between parts of a message
    ".-.-.": "+",            # AR, end of message
    "...-.-": "<SK>",        # end of contact
    "-.--.": "<KN>",         # over, to the named station only
    "........": "<HH>",      # error, correction follows
}
TO_CODE = {v: k for k, v in CODE.items()}


def tone_and_purity(x, fs):
    """Strongest tone in TONE_BAND, and the share of 200-3500 Hz power within +-40 Hz of it."""
    f, p = welch(x, fs=fs, nperseg=2048)
    band = (f >= TONE_BAND[0]) & (f <= TONE_BAND[1])
    k = np.flatnonzero(band)[np.argmax(p[band])]
    near = np.abs(f - f[k]) <= 40
    audio = (f >= 200) & (f <= 3500)
    return float(f[k]), float(p[near].sum() / max(p[audio].sum(), 1e-20))


def envelope(x, fs, tone):
    """Beep loudness at ENV_RATE values per second."""
    block = int(round(fs / ENV_RATE))
    n = len(x) // block * block
    t = np.arange(n) / fs
    y = x[:n] * np.exp(-2j * np.pi * tone * t)
    return np.abs(y.reshape(-1, block).mean(axis=1)).astype(np.float32)


def key_track(env):
    """Envelope -> (key down bool array, snr dB). Two thresholds, carried forward in between."""
    noise, peak = np.percentile(env, 20), np.percentile(env, 99.5)
    snr_db = float(20 * np.log10(max(peak, 1e-12) / max(noise, 1e-12)))
    if snr_db < MIN_SNR_DB:
        return np.zeros(len(env), bool), snr_db
    hi, lo = noise + 0.5 * (peak - noise), noise + 0.3 * (peak - noise)
    state = np.where(env > hi, 1, np.where(env < lo, 0, -1))
    state[0] = max(state[0], 0)
    idx = np.where(state >= 0, np.arange(len(state)), 0)
    np.maximum.accumulate(idx, out=idx)
    return state[idx].astype(bool), snr_db


def runs(keyed):
    """(values, starts, lengths) of the runs in a bool array."""
    if len(keyed) == 0:
        return np.zeros(0, bool), np.zeros(0, int), np.zeros(0, int)
    edges = np.flatnonzero(np.diff(keyed.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(keyed)])
    return keyed[starts], starts, lengths


def clean(keyed, short=GLITCH):
    """Drop key-down blips, then fill key-up gaps, shorter than `short` steps."""
    keyed = keyed.copy()
    for value in (True, False):
        v, s, n = runs(keyed)
        for start, length in zip(s[(v == value) & (n < short)], n[(v == value) & (n < short)]):
            keyed[start:start + length] = not value
    return keyed


def dot_length(ons):
    """Seconds per dot from the key-down times of one transmission."""
    ons = np.asarray(ons, float)
    if len(ons) == 0:
        return 0.06
    lg = np.log(ons)
    c = np.percentile(lg, [20, 80]).astype(float)
    for _ in range(10):
        short = np.abs(lg - c[0]) <= np.abs(lg - c[1])
        if short.all() or (~short).all():
            break
        c = np.array([lg[short].mean(), lg[~short].mean()])
    if np.exp(c[1] - c[0]) > 2:
        dot = float(np.exp(c[0]))
    else:                                   # one group only: all dots, or all dashes
        med = float(np.median(ons))
        dot = med if med < 0.12 else med / 3
    return float(np.clip(dot, 0.025, 0.3))


def decode_runs(values, lengths_s, dot):
    """One transmission's runs (starting with key-down) -> text."""
    out, sym = [], ""
    for down, d in zip(values, lengths_s):
        if down:
            if d >= CARRIER_S:
                if sym:
                    out.append(CODE.get(sym, "*")); sym = ""
                out.append(" [carrier] ")
            else:
                sym += "." if d < 2 * dot else "-"
        else:
            if d >= 2 * dot and sym:
                out.append(CODE.get(sym, "*")); sym = ""
            if d >= 5 * dot:
                out.append(" ")
    if sym:
        out.append(CODE.get(sym, "*"))
    return " ".join("".join(out).split())


def decode_keyed(keyed, tone_at=None, snr_at=None):
    """Whole-file key track -> list of transmissions {start_s, end_s, wpm, text, ...}."""
    keyed = clean(keyed)
    v, s, n = runs(keyed)
    gap = int(SEGMENT_GAP_S * ENV_RATE)
    breaks = np.flatnonzero((~v) & (n >= gap))
    segments, lo = [], 0
    for b in list(breaks) + [len(v)]:
        part = slice(lo, b)
        lo = b + 1
        pv, ps, pn = v[part], s[part], n[part]
        if not pv.any():
            continue
        first = np.argmax(pv)
        pv, ps, pn = pv[first:], ps[first:], pn[first:]
        if pv[-1] == 0:
            pv, ps, pn = pv[:-1], ps[:-1], pn[:-1]
        ons = pn[pv] / ENV_RATE
        if len(ons) < 2:
            continue
        dot = dot_length(ons[ons < CARRIER_S] if (ons < CARRIER_S).any() else ons)
        text = decode_runs(pv, pn / ENV_RATE, dot)
        if not text.strip():
            continue
        start, end = ps[0] / ENV_RATE, (ps[-1] + pn[-1]) / ENV_RATE
        chars = [c for c in text if c != " "]
        seg = {"start_s": round(float(start), 2), "end_s": round(float(end), 2),
               "wpm": round(1.2 / dot, 1), "text": text,
               "unreadable": round(chars.count("*") / max(len(chars), 1), 3)}
        seg["noise_like"] = noise_like(seg["text"], seg["wpm"])
        chunk = int(start // CHUNK_S)
        if tone_at is not None and chunk < len(tone_at):
            seg["tone_hz"] = round(float(tone_at[chunk]), 1)
        if snr_at is not None and chunk < len(snr_at):
            seg["snr_db"] = round(float(snr_at[chunk]), 1)
        segments.append(seg)
    return segments


def noise_like(text, wpm):
    """Noise keyed as Morse decodes as fast strings of E, T, I."""
    chars = [c for c in text if c.isalnum()]
    if wpm > MAX_WPM:
        return True
    return len(chars) >= 6 and sum(c in "ETI" for c in chars) / len(chars) > 0.5


def analyse(info: WavInfo, max_s=None, progress=None):
    """Whole recording -> activity every CHUNK_S + decoded transmissions."""
    tones, purity, snrs, active, keyed = [], [], [], [], []
    for start, x in iter_chunks(info, chunk_s=CHUNK_S, max_s=max_s):
        if len(x) < info.sample_rate:                       # last scrap under 1 s
            continue
        tone, pur = tone_and_purity(x, info.sample_rate)
        k, snr = key_track(envelope(x, info.sample_rate, tone))
        if pur < MIN_PURITY:                                # no beep here, only noise or speech
            k = np.zeros_like(k)
        tones.append(tone); purity.append(pur); snrs.append(snr); active.append(float(k.mean()))
        keyed.append(k)
        if progress:
            progress(start)
    keyed = np.concatenate(keyed) if keyed else np.zeros(0, bool)
    segments = decode_keyed(keyed, tones, snrs)
    kept = [s for s in segments if not s.pop("noise_like")]
    on = np.asarray(active) > 0.01
    return {
        "chunk_s": CHUNK_S,
        "chunks": {"tone_hz": [round(t, 1) for t in tones], "snr_db": [round(s, 1) for s in snrs],
                    "key_down": [round(a, 3) for a in active], "tone_purity": [round(p, 3) for p in purity]},
        "active_minutes": round(float(on.sum()) * CHUNK_S / 60, 1),
        "tone_purity": round(float(np.median(np.asarray(purity)[on])) if on.any() else 0.0, 3),
        "segments": kept,
        "dropped": len(segments) - len(kept),
    }


def synth(text, wpm=20.0, tone=700.0, fs=8000, snr_db=20.0, rng=None, lead_s=0.5):
    """Test / demo signal: `text` keyed at `wpm` on a `tone` beep, in white noise."""
    rng = np.random.default_rng(0) if rng is None else rng
    dot = 1.2 / wpm
    key = [0] * int(lead_s / dot)
    for wi, word in enumerate(text.split()):
        if wi:
            key += [0] * 6                              # 1 after the element + 6 = 7 between words
        for ci, ch in enumerate(word):
            if ci:
                key += [0] * 2                          # 1 after the element + 2 = 3 between letters
            for ei, el in enumerate(TO_CODE[ch]):
                key += [1] * (1 if el == "." else 3) + [0]
    key += [0] * int(lead_s / dot)
    gate = np.repeat(np.asarray(key, float), int(round(dot * fs)))
    ramp = int(0.004 * fs)                              # 4 ms rise/fall, as a real keyer
    gate = np.convolve(gate, np.ones(ramp) / ramp, mode="same")
    t = np.arange(len(gate)) / fs
    sig = gate * np.sin(2 * np.pi * tone * t)
    noise_amp = np.sqrt(0.5 / 10 ** (snr_db / 10))    # SNR per sample, over the full 4 kHz audio band
    return 0.5 * (sig + rng.normal(0, noise_amp, len(sig)))
