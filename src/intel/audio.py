"""Chunked reading of PCM WAV files without loading them whole.

The recorder files this is for are mono 8 kHz 8-bit PCM and up to ~29 h long
(~0.8 GB), so samples are memory-mapped and handed out a chunk at a time as
float32 in [-1, 1]. Plain PCM (8/16/32-bit integer, 32-bit float) only; any
other WAV (ADPCM, mu-law, ...) is refused with a hint to convert it with
ffmpeg first.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    bits: int
    fmt: int            # 1 = integer PCM, 3 = IEEE float
    data_offset: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate


def wav_info(path) -> WavInfo:
    path = Path(path)
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path.name}: not a RIFF/WAVE file")
        fmt = None
        while True:
            head = f.read(8)
            if len(head) < 8:
                raise ValueError(f"{path.name}: no data chunk")
            cid, size = head[:4], struct.unpack("<I", head[4:])[0]
            if cid == b"fmt ":
                body = f.read(size + (size & 1))
                tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:          # WAVE_FORMAT_EXTENSIBLE
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, ch, sr, bits)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path.name}: data before fmt")
                tag, ch, sr, bits = fmt
                if tag not in (1, 3) or bits not in (8, 16, 32):
                    raise ValueError(f"{path.name}: WAV format {tag}/{bits}-bit is not plain PCM; "
                                     f"convert first: ffmpeg -i in.wav -ac 1 -c:a pcm_s16le out.wav")
                offset = f.tell()
                avail = path.stat().st_size - offset
                frames = min(size, avail) // (ch * bits // 8)   # recorders that crash leave size wrong
                return WavInfo(path, sr, ch, bits, tag, offset, frames)
            else:
                f.seek(size + (size & 1), 1)


def _dtype(info: WavInfo):
    if info.fmt == 3:
        return np.float32
    return {8: np.uint8, 16: np.int16, 32: np.int32}[info.bits]


def iter_chunks(info: WavInfo, chunk_s: float = 60.0, max_s: float | None = None):
    """Yields (start_s, float32 mono samples) chunk by chunk."""
    raw = np.memmap(info.path, dtype=_dtype(info), mode="r", offset=info.data_offset,
                    shape=(info.n_frames * info.channels,))
    n = info.n_frames if max_s is None else min(info.n_frames, int(max_s * info.sample_rate))
    step = int(chunk_s * info.sample_rate)
    for s in range(0, n, step):
        e = min(n, s + step)
        x = np.asarray(raw[s * info.channels:e * info.channels])
        if info.channels > 1:
            x = x.reshape(-1, info.channels).mean(axis=1)
        x = x.astype(np.float32)
        if info.fmt == 1:
            if info.bits == 8:
                x = (x - 128.0) / 128.0
            else:
                x /= float(2 ** (info.bits - 1))
        yield s / info.sample_rate, x


def write_wav(path, x, sample_rate, bits=16):
    """Small helper for tests and clips: mono float [-1, 1] -> PCM WAV."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1, 1)
    if bits == 8:
        data = np.round(x * 127 + 128).astype(np.uint8).tobytes()
    else:
        data = np.round(x * 32767).astype("<i2").tobytes()
        bits = 16
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
                      sample_rate, sample_rate * bits // 8, bits // 8, bits, b"data", len(data))
    Path(path).write_bytes(hdr + data)
