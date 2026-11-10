"""
Military-voice radio dataset (English, Malay, Chinese) for the analog / voice
path of the pipeline. Demo and validation only -- never training data for the
60-class model (synthetic-only training rule; voice is handled by the zoom
rules + Whisper + Ollama).

Sources
  Malaysian English  Common Voice cv26-malaysian-english: the clips whose
                     sentence is about armies, navies, troops or radio
  Malay              Common Voice 27.0 ms: the three military sentences, every
                     speaker (validated and not-yet-validated clips, tagged);
                     Spontaneous Speech 5.0 ms-MY: the "military officer" answer
  English, Chinese   radio-procedure lines (fictional call signs) spoken by the
                     offline Windows voices (David, Zira, Huihui)
No Malay radio-procedure lines: Windows has no offline Malay voice.

Each clip -> radio sound (300-3000 Hz, compression, squelch burst, roger
beep) -> one analog transmission (NBFM ground net / AM air band / USB HF,
seeded mix) at a random offset within +-80 kHz and 12-30 dB SNR (3 kHz band)
-> 2-channel int16 IQ WAV at 250 kS/s. scripts/run_pipeline.py --iq reads it
and resamples to 3.2 MS/s (band edges filled with noise).

    python scripts/make_military_voice_dataset.py
    python scripts/run_pipeline.py --iq data/local/military_voice_v1/iq/zh_tts_001.wav \
        --max-seconds 8 --hybrid --classifier results/classifier/physics_aug_best.pt --listen

Output (--out, default data/local/military_voice_v1/, git-ignored), one folder
per language (english/, malay/, chinese/), each with:
  iq/<id>_<MOD>.wav           the RF capture (2-channel IQ, 250 kS/s) -> run_pipeline --iq
  radio_audio/<id>.wav        16 kHz, what the walkie-talkie speaker plays
  clean_audio/<id>.wav        16 kHz speech as recorded / spoken
  manifest.csv                id, source, licence, text, English gloss, modulation,
                              offset, SNR, timing (opens in Excel)
and at the top: manifest.csv / manifest.json (all items) and README.txt.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402

AUDIO_FS = 16000
FA = 50e3
FS_OUT = 250e3
MAX_S = 10.0

DL = Path.home() / "Downloads"


def corpus_dirs(root):
    """The three Common Voice downloads under root (the extracted folders,
    found by name so a different download number still works)."""
    root = Path(root)

    def one(pattern, inner):
        hits = sorted(p for p in root.glob(pattern) if p.is_dir())       # not the .tar.gz next to it
        if not hits:
            sys.exit(f"{root}: no folder matching {pattern} (extract the Common Voice download there, or pass --corpora)")
        d = hits[-1] / inner
        return d if d.exists() else hits[-1]
    return (one("*cv26-malaysian-english*", ""),
            one("*cv-corpus-27.0-*-ms", "cv-corpus-27.0-2026-09-11/ms"),
            one("*sps-corpus-5.0-*-ms-MY", "sps-corpus-5.0-2026-09-11-ms-MY"))


CV_EN = CV_MS = SPS_MS = None             # set in main() from --corpora

# Hand-checked: keyword hits that are really military (not "mission station", "Hot FM" ...).
EN_CLIPS = ["common_voice_en_17665991.mp3", "common_voice_en_18978257.mp3", "common_voice_en_18998747.mp3",
            "common_voice_en_19423957.mp3", "common_voice_en_19642055.mp3", "common_voice_en_20220072.mp3",
            "common_voice_en_23149605.mp3", "common_voice_en_23149645.mp3", "common_voice_en_24397402.mp3",
            "common_voice_en_29492074.mp3", "common_voice_en_37499030.mp3", "common_voice_en_37647902.mp3",
            "common_voice_en_39711244.mp3", "common_voice_en_681643.mp3"]
MS_SENTENCES = {
    "Tentera udara menggunakan dengan mengawal fungsi senjata dan mempertingkatkannya":
        "The air force uses it by controlling weapon functions and enhancing them.",
    "Baginda telah mengetuai pasukan tenteranya di hadapan, sebagai seorang panglima tentera":
        "His Majesty led his troops at the front, as a military commander.",
    "Komuniti awal Mandailing dan Rawa ini ialah pelarian dari perang Padri di Sumatera.":
        "These early Mandailing and Rawa communities were refugees from the Padri War in Sumatra.",
}
SPS_CLIPS = {"spontaneous-speech-ms-MY-20147.mp3":
             "Since I was small I have wanted to become a military officer; after finishing school I was accepted ..."}

EN_TTS = [
    "Bravo Two, this is Alpha One. Radio check, over.",
    "Alpha One, Bravo Two. Read you loud and clear, over.",
    "Alpha One, Bravo Two. Two vehicles moving north on route Echo, grid three four five six. Over.",
    "Bravo Two, Alpha One. Roger. Continue to observe and report any change. Out.",
    "Hotel Six, this is Kilo Three. Arrived at rally point, all personnel accounted for. Over.",
    "Kilo Three, Hotel Six. Request resupply of water and batteries at checkpoint Charlie. Over.",
    "Hotel Six, Kilo Three. Roger. Resupply in three zero minutes. Wait, out.",
    "All stations, this is Sierra Base. Weather closing in, visibility poor. Hold position and await orders. Out.",
]
ZH_TTS = [
    ("猎鹰一号，这里是雷霆二号，无线电检查，完毕。", "Falcon One, this is Thunder Two, radio check, over."),
    ("雷霆二号，猎鹰一号，信号清晰，完毕。", "Thunder Two, Falcon One, reading you clearly, over."),
    ("发现两辆车辆沿公路向北行驶，坐标三四五六，完毕。",
     "Two vehicles spotted moving north along the road, grid three four five six, over."),
    ("收到，继续观察，有变化立即报告，完毕。", "Roger, continue to observe, report any change at once, over."),
    ("我部已到达集结点，人员全部到齐，完毕。", "Our unit has reached the rally point, all personnel present, over."),
    ("请求补给饮用水和电池，地点三号检查站，完毕。",
     "Request resupply of drinking water and batteries at checkpoint three, over."),
    ("明白，补给预计三十分钟后到达，完毕。", "Understood, resupply expected in thirty minutes, over."),
    ("各台注意，天气转坏，能见度低，原地待命，完毕。",
     "All stations, weather worsening, visibility low, hold position and stand by, over."),
]
LANG_DIRS = {"en": "english", "ms": "malay", "zh": "chinese"}
README = """Military voice radio dataset (simulated RF, real and synthetic speech) - {n} transmissions
Made by scripts/make_military_voice_dataset.py (sedicAI_NEXA). Demo and validation only, not for training.

english/  {english} items: 8 radio-procedure lines (Windows voices David/Zira) + 14 Common Voice
          Malaysian-English clips with military sentences (armies, navies, troops)
malay/    {malay} items: Common Voice Malay, 3 military sentences read by 12 speakers
          ("not yet validated" = still in Common Voice's other.tsv) + 1 Spontaneous Speech answer
chinese/  {chinese} items: 8 Mandarin radio-procedure lines (Windows voice Huihui)

In every language folder:
  iq/<id>_<MODULATION>.wav   RF capture: 2-channel IQ WAV (I left, Q right), int16, 250 kS/s.
                             One transmission: NBFM (ground walkie-talkie), AM (air band) or
                             USB (HF), at offset_khz from the centre, snr_db_3khz in white noise.
                             Run:  python scripts/run_pipeline.py --iq <file> --max-seconds 12
                                   --hybrid --classifier results/classifier/physics_aug_best.pt --listen
  radio_audio/<id>.wav       the sound the receiver should give back (16 kHz mono)
  clean_audio/<id>.wav       the original speech (16 kHz mono)
  manifest.csv               text, English meaning, source clip, licence, modulation, offset, SNR, timing

Licences: Common Voice clips CC0-1.0; scripted lines generated locally. Call signs are fictional.
"""
CV_LICENCE = "CC0-1.0 (Mozilla Common Voice)"
TTS_LICENCE = "generated locally (Windows SAPI voice)"


# ---------- audio in / out ----------

def write_wav(path, x, fs=AUDIO_FS):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        fs, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample_poly(x, AUDIO_FS, fs) if fs != AUDIO_FS else x


def decode(mp3):
    ffmpeg = shutil.which("ffmpeg") or r"D:\tools\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", str(AUDIO_FS), str(out)],
                       check=True)
        return read_wav(out)


def tts(text, voice, path):
    """Offline Windows voice -> WAV. The text goes through a UTF-8 file so
    Chinese survives the PowerShell command line."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        txt = f.name
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); "
          "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{path}', $fmt); "
          f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    finally:
        Path(txt).unlink(missing_ok=True)
    return read_wav(path)


# ---------- radio sound and RF ----------

def trim(x, thr=0.02):
    on = np.flatnonzero(np.abs(x) > thr * (np.abs(x).max() + 1e-9))
    return x[max(on[0] - 800, 0):on[-1] + 800] if len(on) else x


def radio(x, rng):
    x = trim(x)[:int(MAX_S * AUDIO_FS)]
    x = sosfiltfilt(butter(4, [300, 3000], btype="band", fs=AUDIO_FS, output="sos"), x)
    # Noise gate before compression: phone recordings (Common Voice) carry room
    # noise between words, and compression would lift it to speech level.
    fr = int(0.02 * AUDIO_FS)
    env = np.sqrt(np.convolve(x ** 2, np.ones(fr) / fr, mode="same"))
    floor, loud = np.percentile(env, 20), np.percentile(env, 95)
    gate = np.clip((env - 2 * floor) / (0.25 * loud - 2 * floor + 1e-9), 0.1, 1.0)
    x = x * np.convolve(gate, np.ones(fr) / fr, mode="same")
    x = x / (np.percentile(np.abs(x), 99.5) + 1e-9)
    x = np.tanh(1.5 * x) / np.tanh(1.5)                   # mild mic AGC, as a handheld
    burst = 0.25 * sosfiltfilt(butter(2, [500, 4000], btype="band", fs=AUDIO_FS, output="sos"),
                               rng.standard_normal(int(0.08 * AUDIO_FS)))
    gap = np.zeros(int(0.12 * AUDIO_FS))
    beep = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(int(0.12 * AUDIO_FS)) / AUDIO_FS)
    return np.concatenate([burst, gap, 0.9 * x, gap[:800], beep, np.zeros(int(0.15 * AUDIO_FS))]).astype(np.float32)


def modulate(audio, kind):
    a = resample_poly(audio, int(FA), AUDIO_FS)
    if kind == "NBFM":
        return np.exp(1j * 2 * np.pi * np.cumsum(2500 * a) / FA)
    if kind == "AM":
        return (1 + 0.8 * a).astype(complex)
    if kind == "USB":
        return hilbert(a)
    raise ValueError(kind)


def to_iq(snd, kind, offset, snr_db, rng, lead_s=0.3, tail_s=0.3):
    """Transmission in noise at FS_OUT, with quiet noise before key-up and after."""
    bb = modulate(snd, kind)
    p = np.mean(np.abs(bb) ** 2)
    bb = bb / np.sqrt(p) * np.sqrt(10 ** (snr_db / 10) * 3000 / FS_OUT)     # unit noise power over FS_OUT
    w = resample_poly(bb, int(FS_OUT / FA), 1)
    lead, tail = int(lead_s * FS_OUT), int(tail_s * FS_OUT)
    n = lead + len(w) + tail
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    iq[lead:lead + len(w)] += w * np.exp(2j * np.pi * offset * (lead + np.arange(len(w))) / FS_OUT)
    return iq, lead / FS_OUT, (lead + len(w)) / FS_OUT


def write_iq(path, iq):
    scale = 0.9 / np.abs(np.concatenate([iq.real, iq.imag])).max()
    pcm = np.empty(2 * len(iq), dtype="<i2")
    pcm[0::2] = np.round(iq.real * scale * 32767)
    pcm[1::2] = np.round(iq.imag * scale * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(FS_OUT))
        w.writeframes(pcm.tobytes())


# ---------- source lists ----------

def tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def sources(tmp):
    items = []
    en = {r["path"]: r for name in ("train.tsv", "dev.tsv", "test.tsv") for r in tsv(CV_EN / name)}
    for i, clip in enumerate(EN_CLIPS, 1):
        r = en[clip]
        items.append({"id": f"en_cv_{i:03d}", "language": "en (Malaysian English)", "source": f"cv26-malaysian-english/{clip}",
                      "speaker": r["client_id"][:8], "licence": CV_LICENCE, "status": "validated",
                      "text": r["sentence"], "gloss": r["sentence"],
                      "load": lambda c=clip: decode(next(CV_EN.rglob(c)))})
    status = {r["path"]: "validated" for r in tsv(CV_MS / "validated.tsv")}
    seen, k = set(), 0
    for name in ("validated.tsv", "other.tsv"):
        for r in tsv(CV_MS / name):
            if r["sentence"] in MS_SENTENCES and r["path"] not in seen:
                seen.add(r["path"])
                k += 1
                items.append({"id": f"ms_cv_{k:03d}", "language": "ms", "source": f"cv-corpus-27.0/ms/{r['path']}",
                              "speaker": r["client_id"][:8], "licence": CV_LICENCE,
                              "status": status.get(r["path"], "not yet validated (other.tsv)"),
                              "text": r["sentence"], "gloss": MS_SENTENCES[r["sentence"]],
                              "load": lambda c=r["path"]: decode(CV_MS / "clips" / c)})
    for r in tsv(SPS_MS / "ss-corpus-ms-MY.tsv"):
        if r["audio_file"] in SPS_CLIPS:
            items.append({"id": "ms_sps_001", "language": "ms", "source": f"sps-corpus-5.0/ms-MY/{r['audio_file']}",
                          "speaker": r["client_id"][:8], "licence": CV_LICENCE, "status": "spontaneous answer",
                          "text": r["transcription"], "gloss": SPS_CLIPS[r["audio_file"]],
                          "load": lambda c=r["audio_file"]: decode(next(SPS_MS.rglob(c)))})
    for i, line in enumerate(EN_TTS, 1):
        v = "Microsoft David Desktop" if i % 2 else "Microsoft Zira Desktop"
        items.append({"id": f"en_tts_{i:03d}", "language": "en", "source": "radio-procedure script", "speaker": v,
                      "licence": TTS_LICENCE, "status": "scripted", "text": line, "gloss": line,
                      "load": lambda t=line, v=v, i=i: tts(t, v, str(Path(tmp) / f"en{i}.wav"))})
    for i, (line, gloss) in enumerate(ZH_TTS, 1):
        v = "Microsoft Huihui Desktop"
        items.append({"id": f"zh_tts_{i:03d}", "language": "zh (Mandarin)", "source": "radio-procedure script",
                      "speaker": v, "licence": TTS_LICENCE, "status": "scripted", "text": line, "gloss": gloss,
                      "load": lambda t=line, v=v, i=i: tts(t, v, str(Path(tmp) / f"zh{i}.wav"))})
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpora", default=str(DL), help="folder holding the 3 extracted Common Voice downloads")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "local" / "military_voice_v1"))
    args = p.parse_args()
    global CV_EN, CV_MS, SPS_MS
    CV_EN, CV_MS, SPS_MS = corpus_dirs(args.corpora)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for it in sources(tmp):
            lang_dir = out / LANG_DIRS[it["id"][:2]]
            for sub in ("iq", "radio_audio", "clean_audio"):
                (lang_dir / sub).mkdir(parents=True, exist_ok=True)
            clean = it.pop("load")()
            write_wav(lang_dir / "clean_audio" / f"{it['id']}.wav", clean)
            snd = radio(clean, rng)
            write_wav(lang_dir / "radio_audio" / f"{it['id']}.wav", snd)
            kind = str(rng.choice(["NBFM", "AM", "USB"], p=[0.5, 0.3, 0.2]))
            offset = float(np.round(rng.uniform(-80e3, 80e3), -2))
            snr = float(np.round(rng.uniform(12, 30), 1))
            iq, t0, t1 = to_iq(snd, kind, offset, snr, rng)
            iq_name = f"{it['id']}_{kind}.wav"
            write_iq(lang_dir / "iq" / iq_name, iq)
            it.update({"folder": LANG_DIRS[it["id"][:2]], "iq_file": f"iq/{iq_name}",
                       "modulation": kind, "offset_khz": offset / 1e3, "snr_db_3khz": snr,
                       "fs": FS_OUT, "tx_start_s": round(t0, 3), "tx_end_s": round(t1, 3),
                       "duration_s": round(len(iq) / FS_OUT, 3)})
            rows.append(it)
            print(f"{it['id']:<11} {it['modulation']:<5} {offset / 1e3:+6.1f} kHz {snr:4.1f} dB "
                  f"{it['duration_s']:5.1f} s  {it['text'][:60]}")
    keys = list(rows[0].keys())

    def write_csv(path, rs):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:     # -sig: Excel shows Chinese correctly
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)

    write_csv(out / "manifest.csv", rows)
    for d in LANG_DIRS.values():
        write_csv(out / d / "manifest.csv", [r for r in rows if r["folder"] == d])
    (out / "manifest.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = {d: sum(r["folder"] == d for r in rows) for d in LANG_DIRS.values()}
    (out / "README.txt").write_text(README.format(n=len(rows), **counts), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{len(rows)} items {counts}, {size:.0f} MB in {out}")


if __name__ == "__main__":
    main()
