"""
Simulated military radio traffic for the voice path of the pipeline (demo and
validation only -- never training data, see the synthetic-only training rule).

1. Speech: radio-procedure lines spoken by the offline Windows voices (SAPI,
   System.Speech), plus optional real clips (e.g. a Common Voice MP3, decoded
   with ffmpeg).
2. Radio sound: 300-3000 Hz band, hard compression, key-up squelch burst,
   carrier tail and a roger beep, like a handheld set.
3. Merge: each station is put on its own channel of a 3.2 MS/s capture --
   NBFM (2.5 kHz deviation, ground tactical nets) or AM (air band) -- over
   white noise, and written as a 2-channel int16 IQ WAV that
   scripts/run_pipeline.py --iq reads directly.

    python scripts/make_military_radio.py
    python scripts/make_military_radio.py --extra-clip path/to/clip.mp3 --seconds 4
    python scripts/run_pipeline.py --iq data/local/military_radio/military_net.wav \
        --max-seconds 4 --hybrid --classifier results/classifier/physics_aug_best.pt --listen

Outputs in --out (default data/local/military_radio/, git-ignored):
  <station>_clean.wav, <station>_radio.wav   16 kHz mono, to listen to
  military_net.wav                            the IQ capture
  military_net.json                           truth: station, channel, kind, text
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402

AUDIO_FS = 16000
FS = 3.2e6
FA = 50e3                     # per-channel baseband rate before upsampling to FS

# Fictional call signs; standard voice procedure (radio check, contact report).
STATIONS = [
    {"name": "alpha", "voice": "Microsoft David Desktop", "kind": "NBFM", "offset_hz": -600e3, "snr_db": 25,
     "text": "Bravo Two, this is Alpha One. Radio check, over."},
    {"name": "bravo", "voice": "Microsoft Zira Desktop", "kind": "NBFM", "offset_hz": 350e3, "snr_db": 22,
     "text": "Alpha One, Bravo Two. Two vehicles moving north on route Echo, grid three four five six. Over."},
]
EXTRA_CLIP = (Path.home() / "Downloads" / "1789491044876-cv-corpus-27.0-2026-09-11-ms" / "cv-corpus-27.0-2026-09-11"
              / "ms" / "clips" / "common_voice_ms_41076275.mp3")
EXTRA_TEXT = "Tentera udara menggunakan dengan mengawal fungsi senjata dan mempertingkatkannya"


def write_wav(path, x, fs=AUDIO_FS):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if fs != AUDIO_FS:
        x = resample_poly(x, AUDIO_FS, fs)
    return x


def tts(text, voice, path):
    """Offline Windows speech (System.Speech) -> 16 kHz mono WAV."""
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.Rate = 1; "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return read_wav(path)


def decode(clip):
    ffmpeg = shutil.which("ffmpeg") or r"D:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clip.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(clip), "-ac", "1", "-ar", str(AUDIO_FS), str(out)],
                       check=True)
        return read_wav(out)


def trim(x, thr=0.02):
    on = np.flatnonzero(np.abs(x) > thr * np.abs(x).max())
    return x[max(on[0] - 800, 0):on[-1] + 800] if len(on) else x


def radio(x, rng):
    """Handheld-radio sound: key-up squelch burst, band-limited compressed
    voice, roger beep, squelch tail."""
    x = trim(x)
    x = sosfiltfilt(butter(4, [300, 3000], btype="band", fs=AUDIO_FS, output="sos"), x)
    x = x / (np.abs(x).max() + 1e-9)
    x = np.tanh(3.0 * x) / np.tanh(3.0)                      # mic AGC / clipping
    burst = 0.25 * sosfiltfilt(butter(2, [500, 4000], btype="band", fs=AUDIO_FS, output="sos"),
                               rng.standard_normal(int(0.08 * AUDIO_FS)))
    gap = np.zeros(int(0.12 * AUDIO_FS))
    t = np.arange(int(0.12 * AUDIO_FS)) / AUDIO_FS
    beep = 0.5 * np.sin(2 * np.pi * 1000 * t)
    tail = np.zeros(int(0.15 * AUDIO_FS))
    return np.concatenate([burst, gap, 0.9 * x, gap[:800], beep, tail]).astype(np.float32)


def modulate(audio, kind):
    """16 kHz audio -> complex baseband at FA (carrier on for the whole transmission)."""
    a = resample_poly(audio, int(FA), AUDIO_FS)
    if kind == "NBFM":
        return np.exp(1j * 2 * np.pi * np.cumsum(2500 * a) / FA)
    if kind == "AM":
        return (1 + 0.8 * a).astype(complex)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=4.0, help="capture length (pipeline CPU: ~50 s per second)")
    p.add_argument("--extra-clip", default=str(EXTRA_CLIP), help="real speech clip put on an AM air channel ('' = none)")
    p.add_argument("--extra-text", default=EXTRA_TEXT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "local" / "military_radio"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations = [dict(s) for s in STATIONS]
    for s in stations:
        s["audio"] = tts(s["text"], s["voice"], str((out / f"{s['name']}_clean.wav").resolve()))
    if args.extra_clip and Path(args.extra_clip).exists():
        a = decode(args.extra_clip)
        write_wav(out / "air_clean.wav", a)
        stations.append({"name": "air", "kind": "AM", "offset_hz": 1.1e6, "snr_db": 25, "audio": a,
                         "text": args.extra_text, "source": Path(args.extra_clip).name})

    n = int(args.seconds * FS)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)   # unit-power noise
    up = int(round(FS / FA))
    truth = []
    for i, s in enumerate(stations):
        snd = radio(s["audio"], rng)
        write_wav(out / f"{s['name']}_radio.wav", snd)
        bb = modulate(snd, s["kind"])
        p_sig = np.mean(np.abs(bb) ** 2)
        bb = bb / np.sqrt(p_sig) * np.sqrt(10 ** (s["snr_db"] / 10) * 3000 / FS)   # SNR in a 3 kHz band
        start = int((0.05 + 0.15 * i) * FS)                                       # stagger the key-ups
        w = resample_poly(bb, up, 1)[:max(n - start, 0)]
        w = w * np.exp(2j * np.pi * s["offset_hz"] * (start + np.arange(len(w))) / FS)
        iq[start:start + len(w)] += w
        truth.append({"station": s["name"], "kind": s["kind"], "offset_khz": s["offset_hz"] / 1e3,
                      "snr_db": s["snr_db"], "t_start_s": round(start / FS, 3),
                      "t_end_s": round((start + len(w)) / FS, 3), "full_tx_s": round(len(snd) / AUDIO_FS, 2),
                      "text": s["text"], "source": s.get("source", f"Windows TTS ({s.get('voice')})")})

    scale = 0.9 / np.abs(np.concatenate([iq.real, iq.imag])).max()
    pcm = np.empty(2 * n, dtype="<i2")
    pcm[0::2] = np.round(iq.real * scale * 32767)
    pcm[1::2] = np.round(iq.imag * scale * 32767)
    with wave.open(str(out / "military_net.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(FS))
        w.writeframes(pcm.tobytes())
    (out / "military_net.json").write_text(json.dumps({"fs": FS, "seconds": args.seconds, "stations": truth}, indent=1))
    for t in truth:
        cut = " (cut by capture end)" if t["full_tx_s"] > t["t_end_s"] - t["t_start_s"] + 0.01 else ""
        print(f"{t['station']:<6} {t['kind']:<5} {t['offset_khz']:+7.0f} kHz  {t['t_start_s']:.2f}-{t['t_end_s']:.2f} s"
              f"{cut}  \"{t['text']}\"")
    print(f"Wrote {out / 'military_net.wav'} ({(out / 'military_net.wav').stat().st_size / 1e6:.0f} MB) and the clips")


if __name__ == "__main__":
    main()
