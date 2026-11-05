"""Speech to text with Whisper, locally (pip install openai-whisper; needs ffmpeg
only for formats other than the WAVs read here).

Only the stretches where the audio is loud enough to hold speech are sent,
up to `max_minutes` per recording: Whisper on a CPU runs at about real time,
so hours of receiver hiss would take hours for nothing.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .audio import WavInfo, iter_chunks

_MODELS = {}


def load(name="base"):
    if name not in _MODELS:
        import whisper                     # imported here: optional, and slow to import
        _MODELS[name] = whisper.load_model(name)
    return _MODELS[name]


def active_audio(info: WavInfo, max_minutes=10.0, chunk_s=30.0):
    """The loudest-relative-to-the-file 30 s chunks, in time order, as 16 kHz float32 + their starts."""
    levels, starts = [], []
    for start, x in iter_chunks(info, chunk_s=chunk_s):
        levels.append(float(np.sqrt(np.mean(x ** 2)))); starts.append(start)
    if not levels:
        return np.zeros(0, np.float32), []
    floor = np.percentile(levels, 20)
    keep = [i for i in np.argsort(levels)[::-1] if levels[i] > 2 * floor][: int(max_minutes * 60 / chunk_s)]
    keep = sorted(keep)
    picked = []
    for i, (start, x) in enumerate(iter_chunks(info, chunk_s=chunk_s)):
        if i in keep:
            picked.append(resample_poly(x, 16000, info.sample_rate).astype(np.float32))
    audio = np.concatenate(picked) if picked else np.zeros(0, np.float32)
    return audio, [starts[i] for i in keep]


def transcribe(info: WavInfo, model="base", max_minutes=10.0, language=None):
    audio, starts = active_audio(info, max_minutes)
    if len(audio) == 0:
        return {"text": "", "language": None, "chunks_s": [], "model": model}
    res = load(model).transcribe(audio, language=language, fp16=False)
    return {"text": res.get("text", "").strip(), "language": res.get("language"),
            "chunks_s": starts, "model": model,
            # times inside the clip of loud chunks (chunks_s gives where each chunk sits in the file)
            "segments": [{"clip_start_s": round(s["start"], 1), "clip_end_s": round(s["end"], 1),
                          "text": s["text"].strip()}
                         for s in res.get("segments", [])]}
