"""Simulated noisy radio voice with a known script, so speech-to-text can be scored.

  speak()      a line of text -> WAV, with the voices built into Windows (offline)
  SCENARIOS    speech -> speech plus hiss, radio band-limiting, a whistle, clicks,
               fading, dropouts or a second talker, each at a set level
  wer()        word error rate of a transcript against the script

SNR here is the speech's power while talking against the noise's power across
the whole 0-8 kHz audio band, so 0 dB means equal power, not "half audible":
hiss spreads over 8 kHz while speech sits mostly below 4 kHz.

The speech is a Windows text-to-speech voice: clear and evenly paced, easier than
a person talking on a radio. Scores here are an upper bound for real recordings,
and scripts/simulate_voice.py --speech / --speech-dir takes real recordings instead.
Malay has a script (LINES_MS) but no Windows voice on this laptop, so it runs on
recordings of people reading that script.

numpy and scipy only, so it runs in any of the project's Python environments.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .audio import iter_chunks, wav_info, write_wav

FS = 16000
VOICES = ("Microsoft Zira Desktop", "Microsoft David Desktop")
# Made-up traffic, numbers spoken digit by digit as a radio operator would.
LINES = (
    "Alpha one, this is base. Move to checkpoint seven and report when you arrive. Over.",
    "Base, this is Alpha one. We can see two vehicles near the river bridge. Request permission to continue. Over.",
    "All stations, the weather is getting worse. Heavy rain is expected after sunset, so return to camp before dark.",
    "Bravo two, your signal is weak and broken. Say again your position and change to channel four. Out.",
    "The supply truck will leave the north gate at nine in the morning, carrying food, water and medical equipment.",
    "Please confirm that the generator has been repaired, and tell the team to check the radio batteries tonight.",
)
# The same traffic in Malay. No Malay voice ships with Windows here, so these are
# for reading aloud: scripts/simulate_voice.py --language ms --print-script.
LINES_MS = (
    "Alfa satu, ini pangkalan. Bergerak ke pusat pemeriksaan tujuh dan lapor apabila tiba. Ganti.",
    "Pangkalan, ini Alfa satu. Kami nampak dua kenderaan berhampiran jambatan sungai. Mohon kebenaran untuk teruskan. Ganti.",
    "Semua stesen, cuaca semakin buruk. Hujan lebat dijangka selepas matahari terbenam, jadi pulang ke kem sebelum gelap.",
    "Bravo dua, isyarat anda lemah dan terputus-putus. Ulang semula kedudukan anda dan tukar ke saluran empat. Habis.",
    "Lori bekalan akan bertolak dari pintu utara pada pukul sembilan pagi, membawa makanan, air dan peralatan perubatan.",
    "Sila sahkan bahawa penjana telah dibaiki, dan beritahu pasukan untuk memeriksa bateri radio malam ini.",
)
SCRIPTS = {"en": LINES, "ms": LINES_MS}
_SAPI = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.SelectVoice($env:NEXA_VOICE);"
    "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
    "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
    "$s.SetOutputToWaveFile($env:NEXA_WAV, $f);"
    "[void]$s.Speak($env:NEXA_TEXT);"
    "$s.Dispose()"
)
_DIGITS = {"en": "zero one two three four five six seven eight nine".split(),
           "ms": "kosong satu dua tiga empat lima enam tujuh lapan sembilan".split()}


def recordings(folder, language="en"):
    """Your own readings of the script: line1.wav, line2.wav ... in `folder`.
    -> list of (float32 speech, script line, label). Lines without a file are
    skipped, so you can record only some of them."""
    script = SCRIPTS[language]
    folder = Path(folder)
    found = []
    for i, line in enumerate(script, 1):
        path = folder / f"line{i}.wav"
        if path.exists():
            found.append((read_wav(path), line, f"line{i}_{folder.name}"))
    return found


def common_voice(folder, n=20, seed=0, cache_dir=None, split="validated"):
    """Real people from a Mozilla Common Voice download (MP3 clips + <split>.tsv).
    -> list of (float32 speech, sentence, label cv01, cv02 ...), n clips taken in
    turn from each speaker, so one talkative person does not dominate. Speaker ids
    are only used to spread the picks, and never kept: the licence forbids trying
    to identify speakers, and it forbids re-hosting the data, so keep it local.

    split: "validated" (checked by other volunteers) by default; "other" holds
    many more clips, but nobody has confirmed their text matches the audio."""
    folder = Path(folder)
    tsvs = sorted(folder.rglob(f"{split}.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"no {split}.tsv under {folder}; unpack the Common Voice download there")
    with open(tsvs[0], encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                if r.get("path") and (r.get("sentence") or "").strip()]
    clips_dir = tsvs[0].parent / "clips"
    by_speaker = {}
    for r in rows:
        if (clips_dir / r["path"]).exists():
            by_speaker.setdefault(r.get("client_id", ""), []).append(r)
    rng = np.random.default_rng(seed)
    queues = [list(rng.permutation(np.array(v, dtype=object))) for _, v in sorted(by_speaker.items())]
    rng.shuffle(queues)
    picked = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    cache = Path(cache_dir) if cache_dir else folder / "_wav16k"
    cache.mkdir(parents=True, exist_ok=True)
    found = []
    for i, r in enumerate(picked, 1):
        wav = cache / (Path(r["path"]).stem + ".wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(clips_dir / r["path"]),
                            "-ar", str(FS), "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=True, timeout=120)
        found.append((read_wav(wav), r["sentence"].strip(), f"cv{i:02d}"))
    return found


def speak(text, path, voice=VOICES[0]):
    """Windows text-to-speech -> 16 kHz mono WAV at `path`. Fully local."""
    env = dict(os.environ, NEXA_VOICE=voice, NEXA_WAV=str(path), NEXA_TEXT=text)
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _SAPI],
                   env=env, check=True, capture_output=True, timeout=120)
    return Path(path)


def spoken(text, voice, cache_dir):
    """speak() once per (voice, text), then read it back as float32."""
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{key}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        speak(text, path, voice)
    return read_wav(path)


def read_wav(path):
    """Any PCM WAV -> float32 mono at FS, for speech-to-text."""
    info = wav_info(path)
    x = np.concatenate([c for _, c in iter_chunks(info)]) if info.n_frames else np.zeros(0, np.float32)
    if info.sample_rate != FS:
        x = resample_poly(x, FS, info.sample_rate)
    return np.asarray(x, dtype=np.float32)


def save_wav(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, x, FS, bits=16)


# --- levels ---------------------------------------------------------------------

def active_power(x, frame=320):
    """Mean power of the frames that hold speech (above 1% of the loud frames),
    so pauses do not make the speech look quieter than it is."""
    n = len(x) // frame
    if n == 0:
        return float(np.mean(np.square(x))) if len(x) else 0.0
    p = np.mean(np.square(np.asarray(x[:n * frame], dtype=np.float64)).reshape(n, frame), axis=1)
    return float(p[p > 0.01 * np.percentile(p, 95)].mean()) if p.max() > 0 else 0.0


def hiss(n, power, rng):
    """White noise of the given power across the whole audio band."""
    return rng.standard_normal(n) * np.sqrt(power)


def with_hiss(x, snr_db, rng):
    return x + hiss(len(x), active_power(x) / 10 ** (snr_db / 10), rng)


def _fit(x):
    """Scale down only if it would clip, so the WAV stays honest."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * 0.95 / peak if peak > 0.95 else x).astype(np.float32)


def radio_band(x):
    """What a voice radio does to speech: 300-3400 Hz, then hard compression."""
    y = sosfiltfilt(butter(4, [300, 3400], btype="band", fs=FS, output="sos"), x)
    y = np.tanh(3.0 * y / (np.max(np.abs(y)) + 1e-9))
    return y * np.sqrt(active_power(x) / (active_power(y) + 1e-12))


# --- scenarios: each is f(speech, other_talker, rng) -> noisy --------------------

def _whistle(x, other, rng):
    tone = np.sqrt(2 * active_power(x) * 10 ** 0.6) * np.sin(2 * np.pi * 1000 * np.arange(len(x)) / FS)
    return with_hiss(x, 10, rng) + tone                # tone 6 dB above the speech


def _clicks(x, other, rng):
    y = with_hiss(x, 10, rng)
    rms = np.sqrt(active_power(x))
    for start in rng.integers(0, max(len(x) - 64, 1), int(8 * len(x) / FS)):     # 8 clicks a second
        burst = rng.choice([-1, 1], 32) * np.exp(-np.arange(32) / 8)
        y[start:start + 32] += rng.uniform(3, 6) * rms * burst
    return y


def _fading(x, other, rng):
    t = np.arange(len(x)) / FS
    swing = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 1.0) * t + rng.uniform(0, 6.28))
    hissy = hiss(len(x), active_power(x) / 10 ** 1.5, rng)      # noise stays put, speech fades up to 24 dB
    return x * 10 ** (-24 * swing ** 2 / 20) + hissy


def _dropouts(x, other, rng):
    """Squelch or signal gaps: about 15% of the speech is simply gone."""
    y = x.copy()
    missing, target = 0, int(0.15 * len(x))
    while missing < target:
        n = int(rng.uniform(0.2, 0.5) * FS)
        s = int(rng.integers(0, max(len(x) - n, 1)))
        y[s:s + n] = 0
        missing += n
    return y + hiss(len(x), active_power(x) / 10 ** 1.5, rng)


def _talker(sir_db):
    def f(x, other, rng):
        o = np.resize(other, len(x))
        o = o * np.sqrt(active_power(x) / (active_power(o) + 1e-12) / 10 ** (sir_db / 10))
        return with_hiss(x + o, 20, rng)
    return f


def _noise_only(x, other, rng):
    """No speech at all: anything the decoder writes here is invented."""
    return hiss(len(x), active_power(x), rng)


def _hiss(snr):
    return lambda x, other, rng: with_hiss(x, snr, rng)


def _radio(snr):
    return lambda x, other, rng: with_hiss(radio_band(x), snr, rng)


SCENARIOS = {
    "clean": lambda x, other, rng: x.copy(),
    **{f"hiss_{s:+d}dB": _hiss(s) for s in (20, 10, 5, 0, -5, -10)},
    **{f"radio_{s:+d}dB": _radio(s) for s in (10, 0, -5)},
    "whistle": _whistle,
    "clicks": _clicks,
    "fading": _fading,
    "dropouts": _dropouts,
    "second_voice_+6dB": _talker(6),
    "second_voice_0dB": _talker(0),
    "noise_only": _noise_only,
}
NO_SPEECH = {"noise_only"}


def simulate(name, speech, other, rng):
    """One scenario -> float32 audio, never clipped."""
    x = np.asarray(speech, dtype=np.float64)
    return _fit(np.asarray(SCENARIOS[name](x, np.asarray(other, dtype=np.float64), rng)))


# --- scoring ----------------------------------------------------------------------

def normalize(text, language="en"):
    """Lower case, no punctuation, digits spoken one by one in `language`
    ("Alpha 12" matches "alpha one two", and in Malay "Alfa satu dua")."""
    digits = _DIGITS.get(language, _DIGITS["en"])
    words = []
    for w in re.findall(r"[a-z0-9']+", text.lower().replace("-", " ")):
        w = w.replace("'", "")
        words += [digits[int(c)] for c in w] if w.isdigit() else [w]
    return words


def wer(truth, hyp, language="en"):
    """Word error rate: (substitutions + deletions + insertions) / truth words.
    Returns dict(wer, sub, del, ins, truth_words, hyp_words). Can exceed 1 when
    the decoder invents more words than were said."""
    t, h = normalize(truth, language), normalize(hyp, language)
    # cell = (cost, subs, dels, ins), row by row
    prev = [(j, 0, 0, j) for j in range(len(h) + 1)]
    for i, tw in enumerate(t, 1):
        cur = [(i, 0, i, 0)]
        for j, hw in enumerate(h, 1):
            d = prev[j - 1]
            options = [d if tw == hw else (d[0] + 1, d[1] + 1, d[2], d[3])]
            u = prev[j]
            options.append((u[0] + 1, u[1], u[2] + 1, u[3]))
            le = cur[j - 1]
            options.append((le[0] + 1, le[1], le[2], le[3] + 1))
            cur.append(min(options))
        prev = cur
    cost, sub, dele, ins = prev[-1]
    return {"wer": cost / len(t) if t else float(len(h) > 0), "sub": sub, "del": dele, "ins": ins,
            "truth_words": len(t), "hyp_words": len(h)}
