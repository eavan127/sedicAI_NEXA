"""
Read a recorded IQ capture into the pipeline's form: complex64 at 3.2 MS/s.

Supported:
  .wav             2-channel IQ WAV (I left, Q right) as written by SDRconnect /
                   SDRuno / SDR# / GQRX: 8, 16 or 32-bit integer PCM or 32-bit
                   float. An 'auxi' chunk (SDRuno/SDRconnect) gives the centre
                   frequency; otherwise it is unknown (None).
  .sigmf-meta /    SigMF pair; datatype cf32_le, ci16_le or ci8, sample rate
  .sigmf-data      and centre frequency from the meta file
  .npy             complex array (sample rate from --fs, default 3.2 MS/s)
  anything else    raw interleaved I/Q of `dtype` (float32, int16, int8, uint8)

Other sample rates are resampled to 3.2 MS/s. When the recording is NARROWER
than 3.2 MHz (e.g. an RSPduo at 2 MS/s), the band beyond its edges would be
empty after resampling; the classifier was trained with noise across the whole
band (an empty edge is exactly the RadioML fingerprint we removed from the
training data), so the edges are filled with white noise at the measured
noise floor.
"""
import json
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly

from src.config import CFG

FS = float(CFG["signal"]["fs"])
EDGE_GUARD = 0.9            # treat the outer 10% of a narrower recording's band as filter roll-off


def _wav(path, max_s):
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path.name}: not a WAV file")
        fmt, centre, data_off, data_len = None, None, None, None
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            cid, size = h[:4], struct.unpack("<I", h[4:])[0]
            if cid == b"fmt ":
                body = f.read(size + (size & 1))
                tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, ch, sr, bits)
            elif cid == b"auxi":
                body = f.read(size + (size & 1))
                if len(body) >= 36:                       # 2 SYSTEMTIMEs (32 bytes), then centre Hz
                    centre = float(struct.unpack("<I", body[32:36])[0]) or None
            elif cid == b"data":
                data_off, data_len = f.tell(), size
                break
            else:
                f.seek(size + (size & 1), 1)
    if fmt is None or data_off is None:
        raise ValueError(f"{path.name}: WAV without fmt or data chunk")
    tag, ch, sr, bits = fmt
    if ch != 2:
        raise ValueError(f"{path.name}: {ch}-channel WAV. An IQ recording has 2 channels (I and Q); "
                         f"this is probably demodulated AUDIO -- in SDRconnect set Recording type = IQ")
    dtype = {(1, 8): np.uint8, (1, 16): np.int16, (1, 32): np.int32, (3, 32): np.float32}.get((tag, bits))
    if dtype is None:
        raise ValueError(f"{path.name}: unsupported WAV sample format (tag {tag}, {bits}-bit)")
    avail = (path.stat().st_size - data_off) // (2 * bits // 8)
    frames = min(data_len // (2 * bits // 8) if data_len else avail, avail)
    if max_s:
        frames = min(frames, int(max_s * sr))
    raw = np.fromfile(path, dtype=dtype, count=2 * frames, offset=data_off).astype(np.float32)
    if dtype == np.uint8:
        raw = (raw - 128) / 128
    elif dtype != np.float32:
        raw /= float(2 ** (bits - 1))
    return raw[0::2] + 1j * raw[1::2], float(sr), centre, f"wav {bits}-bit{' float' if tag == 3 else ''}"


def _sigmf(path, max_s):
    meta_p = path.with_suffix(".sigmf-meta")
    data_p = path.with_suffix(".sigmf-data")
    meta = json.loads(meta_p.read_text())
    g = meta["global"]
    dt = g["core:datatype"]
    sr = float(g["core:sample_rate"])
    caps = meta.get("captures") or [{}]
    centre = caps[0].get("core:frequency")
    dtype = {"cf32_le": np.float32, "ci16_le": np.int16, "ci8": np.int8, "cu8": np.uint8}.get(dt)
    if dtype is None:
        raise ValueError(f"{meta_p.name}: datatype {dt} not supported")
    count = -1 if not max_s else 2 * int(max_s * sr)
    raw = np.fromfile(data_p, dtype=dtype, count=count).astype(np.float32)
    if dtype == np.int16:
        raw /= 32768
    elif dtype == np.int8:
        raw /= 128
    elif dtype == np.uint8:
        raw = (raw - 128) / 128
    return raw[0::2] + 1j * raw[1::2], sr, (float(centre) if centre else None), f"sigmf {dt}"


def _raw(path, dtype, max_s, fs):
    dt = np.dtype(dtype)
    count = -1 if not max_s else 2 * int(max_s * fs)
    raw = np.fromfile(path, dtype=dt, count=count).astype(np.float32)
    if dt == np.int16:
        raw /= 32768
    elif dt == np.int8:
        raw /= 128
    elif dt == np.uint8:
        raw = (raw - 128) / 128
    return raw[0::2] + 1j * raw[1::2], fs, None, f"raw {dt.name}"


def to_fs(iq, fs_in, rng=None):
    """Resample to 3.2 MS/s; refill the band edges a narrower recording leaves empty."""
    iq = np.asarray(iq, dtype=np.complex64)
    if abs(fs_in - FS) < 1:
        return iq
    ratio = Fraction(FS / fs_in).limit_denominator(1000)
    y = resample_poly(iq, ratio.numerator, ratio.denominator).astype(np.complex64)
    if fs_in < FS:
        # Single precision and in place: a 10 s recording is 34 M samples at
        # 3.2 MS/s, and the double-precision version (fftfreq, two FFTs and a
        # np.where copy, ~550 MB each) ran out of memory on an 8 GB laptop.
        rng = rng or np.random.default_rng(0)
        n = len(y)
        kc = EDGE_GUARD * fs_in / 2 * n / FS          # bins with |f| below this are the recorded band
        a, b = int(np.ceil(kc)), int(np.floor(n - kc)) + 1   # Y[a:b] lies outside it
        Y = sfft.fft(y, overwrite_x=True)             # complex64 in, complex64 out
        # Noise floor per bin: the median bin inside the recorded band is noise.
        # The power of a complex-Gaussian bin is exponential, whose median is
        # ln 2 of its mean, so the median alone left the fill 1.6 dB quieter
        # than the real noise: a step at the band edge for the detector to see.
        floor = np.median(np.abs(np.concatenate([Y[:a], Y[b:]])) ** 2) / np.log(2)
        m = max(b - a, 0)
        # White noise has independent complex-Gaussian bins, so draw them
        # directly (E|W|^2 = floor) instead of transforming a noise signal.
        scale = np.float32(np.sqrt(floor / 2))
        Y[a:b] = (rng.standard_normal(m, dtype=np.float32) * scale
                  + 1j * (rng.standard_normal(m, dtype=np.float32) * scale))
        y = sfft.ifft(Y, overwrite_x=True).astype(np.complex64, copy=False)
    return y


def load(path, fs=None, dtype="float32", max_s=None, rng=None):
    """path -> (iq complex64 at 3.2 MS/s, info dict)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        iq, sr, centre, kind = _wav(path, max_s)
    elif suffix in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        iq, sr, centre, kind = _sigmf(path, max_s)
    elif suffix == ".npy":
        sr = float(fs or FS)
        iq = np.load(path).astype(np.complex64)
        if max_s:
            iq = iq[:int(max_s * sr)]
        centre, kind = None, "npy complex"
    else:
        sr = float(fs or FS)
        iq, sr, centre, kind = _raw(path, dtype, max_s, sr)
    if fs and suffix in (".wav",) and abs(fs - sr) > 1:
        sr = float(fs)                                   # trust the user over a wrong header
    out = to_fs(iq, sr, rng)
    info = {"file": path.name, "format": kind, "fs_in": sr, "centre_hz": centre,
            "duration_s": round(len(out) / FS, 4), "resampled": abs(sr - FS) >= 1,
            "edge_filled": sr < FS}
    return out, info
