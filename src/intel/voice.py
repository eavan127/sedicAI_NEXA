"""Speech to text with Whisper, locally (pip install openai-whisper; needs ffmpeg
only for formats other than the WAVs read here).

Only the stretches where the audio is loud enough to hold speech are sent,
up to `max_minutes` per recording: Whisper on a CPU is slower than real time,
so hours of receiver hiss would take hours for nothing.

The default is the `medium` model. On the same 17 Common Voice clips it made 11%
word errors on clean Malay and 10% on Malaysian English, against 48% and 26% for
`base` (scripts/simulate_voice.py, 2026-11-07), at about 6 s of CPU per second of
audio. A missing model is an error, never a download: this stays offline.

Whisper's own confidence is a weak warning: `medium` marked nothing as doubtful
on Malay under light hiss, with 25% of the words wrong. A segment without a doubt
mark is not known to be right, so show the audio next to the text.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .audio import WavInfo, iter_chunks

DEFAULT_MODEL = "medium"
LANGUAGES = ("auto", "ms", "en", "id", "zh")  # what the pipeline and dashboard offer; auto = let Whisper detect
LOW_LOGPROB, HIGH_NO_SPEECH = -1.0, 0.6     # beyond these a segment counts as doubtful
_MODELS = {}


def model_dir() -> Path:
    """Where Whisper keeps its models (its own rule, so the two always agree)."""
    return Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "whisper"


def model_for_this_pc():
    """The Whisper model to use when none is asked for: DEFAULT_MODEL (medium) needs
    3-5 GB free and crashes Python outright when it does not get it, so a PC with
    under 12 GB of RAM uses small (13 Nov 2026: medium segfaulted on a 7.3 GB laptop)."""
    have = installed_models()
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 2 ** 30
    except Exception:                       # noqa: BLE001 -- unknown: keep the default
        ram_gb = 99.0
    if ram_gb < 12 and "small" in have:
        return "small"
    return DEFAULT_MODEL if DEFAULT_MODEL in have or not have else have[-1]


def installed_models():
    """Names of the models on this machine, e.g. ['base', 'medium']."""
    d = model_dir()
    return sorted(p.stem for p in d.glob("*.pt")) if d.is_dir() else []


def load(name=DEFAULT_MODEL):
    if name not in _MODELS:
        path = model_dir() / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Whisper model '{name}' is not on this machine ({path}), and this never downloads by itself. "
                f"While online, run once: python -c \"import whisper; whisper.load_model('{name}')\"")
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
    max_n = max(1, int(max_minutes * 60 / chunk_s))
    if len(levels) <= max_n:                    # short clip (e.g. one demodulated track): all of it
        keep = list(range(len(levels)))
    else:
        floor = np.percentile(levels, 20)
        keep = [i for i in np.argsort(levels)[::-1] if levels[i] > 2 * floor][:max_n]
    keep = sorted(keep)
    picked = []
    for i, (start, x) in enumerate(iter_chunks(info, chunk_s=chunk_s)):
        if i in keep:
            picked.append(resample_poly(x, 16000, info.sample_rate).astype(np.float32))
    audio = np.concatenate(picked) if picked else np.zeros(0, np.float32)
    return audio, [starts[i] for i in keep]


def detect_language(audio, model=DEFAULT_MODEL, among=None):
    """The spoken language of 16 kHz audio, chosen only among `among` (default: the
    languages NEXA offers, LANGUAGES without 'auto'). Whisper's free guess often
    lands on a neighbour (Malay heard as Indonesian, Javanese, Tagalog ...); limited
    to the supported set it is far steadier. -> (code, {code: probability})."""
    import whisper
    among = [l for l in (among or LANGUAGES) if l != "auto"]
    m = load(model)
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(np.asarray(audio, np.float32)),
                                      n_mels=m.dims.n_mels).to(m.device)
    _, probs = m.detect_language(mel)
    p = {l: float(probs.get(l, 0.0)) for l in among}
    total = sum(p.values()) or 1.0
    p = {l: round(v / total, 3) for l, v in p.items()}
    return max(p, key=p.get), p


def transcribe(info: WavInfo, model=DEFAULT_MODEL, max_minutes=10.0, language=None):
    """language: 'ms', 'en', ... or None / 'auto' to detect. Setting it is safer on
    short clips, but a wrong setting gives wrong-language nonsense, not an error."""
    language = None if language in (None, "", "auto") else language
    audio, starts = active_audio(info, max_minutes)
    if len(audio) == 0:
        return {"text": "", "language": language, "chunks_s": [], "model": model, "segments": [], "doubtful_share": 0.0}
    detected = None
    if language is None:                       # "detect": Whisper picks, among the supported languages only
        language, detected = detect_language(audio, model)
    # Each segment decoded alone: feeding text forward can repeat one invented phrase.
    res = load(model).transcribe(audio, language=language, fp16=False, condition_on_previous_text=False)
    segments, doubtful_words, words = [], 0, 0
    for s in res.get("segments", []):
        text = s["text"].strip()
        doubtful = bool(s.get("avg_logprob", 0.0) < LOW_LOGPROB or s.get("no_speech_prob", 0.0) > HIGH_NO_SPEECH)
        n = len(text.split())
        words += n
        doubtful_words += n if doubtful else 0
        # times inside the clip of loud chunks (chunks_s gives where each chunk sits in the file)
        segments.append({"clip_start_s": round(s["start"], 1), "clip_end_s": round(s["end"], 1),
                         "text": text, "doubtful": doubtful})
    return {"text": res.get("text", "").strip(), "language": res.get("language"),
            "chunks_s": starts, "model": model, "segments": segments, "language_detected": detected,
            "doubtful_share": round(doubtful_words / words, 3) if words else 0.0}
