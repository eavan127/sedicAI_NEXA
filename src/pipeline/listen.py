"""
"What does it say?": the last stage of the pipeline, for voice and Morse tracks.

For each track the classifier puts near the analog / tone classes:
  1. cut the track out of the capture (its whole time span, its band)
  2. zoom in (50 kS/s; 400 kS/s for broadcast FM) and name the analog kind
     with measurements (src/pipeline/analog.py): NBFM, AM, USB, LSB, CW, ...
  3. demodulate to 16 kHz audio (src/pipeline/demod.py) and save a WAV,
     so the dashboard can play it
  4. CW -> Morse decoder (src/intel/morse.py); speech -> Whisper
     (src/intel/voice.py, pretrained, local), only for tracks >= MIN_SPEECH_S
  5. text -> local Ollama reading, grounded in that text (src/intel/reason.py)

Every step is optional and fails soft: a missing Whisper or Ollama leaves the
audio and the analog kind in place, with the error noted.
"""
from pathlib import Path

import numpy as np

from src.config import CLASS_GROUPS, CFG
from src.intel import audio as wavio
from src.pipeline import analog, demod, extract

FS = float(CFG["signal"]["fs"])
LISTEN_CLASSES = set(CLASS_GROUPS["ANALOG"]) | {"RADAR_CW", "JAM_TONE", "OOK", "JAM_SPOT"}
TOP5_ANALOG_P = 0.2
MAX_WIDTH_HZ = 300e3            # wider tracks are not voice/Morse radios
MIN_SPEECH_S = 1.0
WIDE_DECIM = 8                  # first zoom: 400 kS/s, wide enough for broadcast FM and loose boxes
VOICE_KINDS = {"NBFM", "WBFM", "AM", "AM_SC", "USB", "LSB"}


def wanted(ans):
    """Should this track be zoomed into?"""
    width = (ans["f_hi_khz"] - ans["f_lo_khz"]) * 1e3
    if width > MAX_WIDTH_HZ:
        return False
    if LISTEN_CLASSES & set(ans.get("classes") or [ans["answer"]]):
        return True
    return any(c in CLASS_GROUPS["ANALOG"] and p >= TOP5_ANALOG_P for c, p in ans.get("top5", []))


def listen_track(iq, ans, out_dir, noise_var, rng, whisper_model="base", llm_model=None,
                 use_whisper=True, use_llm=True, language=None):
    t0 = int(ans["t_start_ms"] * FS / 1e3)
    t1 = int(ans["t_end_ms"] * FS / 1e3)
    f_lo, f_hi = ans["f_lo_khz"] * 1e3, ans["f_hi_khz"] * 1e3
    c = extract.cut(iq, t0, t1, f_lo, f_hi, noise_var, rng)
    # Two-step zoom: 400 kS/s first, find the signal itself inside the (often
    # much wider) box, then centre on it and zoom to 50 kS/s for voice / Morse.
    fa1 = FS / WIDE_DECIM
    y1 = analog.zoom(c.filled, WIDE_DECIM)          # noise-filled outside the box: the floor stays measurable
    m1 = analog.measure(y1, fa1)
    lo1, hi1 = m1.get("band", (-fa1 / 4, fa1 / 4))
    wide = ans["answer"] == "WBFM" or (hi1 - lo1) > analog.WBFM_HZ
    if wide:
        fa, y = fa1, analog._bandpass(y1, fa1, lo1 - 10e3, hi1 + 10e3)
    else:
        centre = m1["carrier_hz"] if m1.get("carrier_frac", 0) >= analog.CARRIER_FRAC else (lo1 + hi1) / 2
        y1 = y1 * np.exp(-2j * np.pi * centre * np.arange(len(y1)) / fa1)
        fa, y = FS / analog.DECIM, analog.zoom(y1, analog.DECIM // WIDE_DECIM)
    kind, feats, why = analog.classify(y, fa, wide_hint=wide)
    out = {"kind": kind, "why": why, "duration_s": round(len(y) / fa, 3),
           "features": {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                        for k, v in feats.items() if k != "band"}}
    if kind == "UNKNOWN":
        return out
    sound = demod.demodulate(y, fa, kind, feats)
    if sound is None:
        return out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"track_{ans['track']}.wav"
    wavio.write_wav(wav, sound, demod.AUDIO_FS, bits=16)
    out["audio"] = wav.name
    info = wavio.wav_info(wav)
    segments, text_kind = [], None
    try:
        if kind == "CW":
            from src.intel import morse
            m = morse.analyse(info)
            segments = m["segments"]
            out["morse"] = " / ".join(s["text"] for s in segments)
            out["morse_detail"] = m
            text_kind = "morse"
        elif kind in VOICE_KINDS and use_whisper and out["duration_s"] >= MIN_SPEECH_S:
            from src.intel import voice
            tr = voice.transcribe(info, model=whisper_model, max_minutes=1.0, language=language)
            out["transcript"] = tr.get("text", "")
            out["language"] = tr.get("language")
            segments = [{"start_s": s["clip_start_s"], "end_s": s["clip_end_s"], "text": s["text"]}
                        for s in tr.get("segments", [])]
            out["speech_segments"] = segments
            text_kind = "voice"
    except Exception as e:                       # noqa: BLE001 -- keep the audio even if decoding fails
        out["text_error"] = f"{type(e).__name__}: {e}"
    if segments and use_llm:
        from src.intel import reason
        kw = {"model": llm_model} if llm_model else {}
        out["reading"] = reason.read_recording(f"track {ans['track']}", segments, kind=text_kind,
                                               duration=f"{out['duration_s']:.1f} s",
                                               active_min=out.get("morse_detail", {}).get("active_minutes", 0), **kw)
    return out


def groups(answers, min_overlap=0.5):
    """Wanted tracks on the same frequency, joined: Morse between words, or
    speech between pauses, is cut into several tracks by the tracker, but the
    operator hears one transmission. -> list of lists of answers."""
    out = []
    for a in sorted((a for a in answers if wanted(a)), key=lambda a: a["t_start_ms"]):
        for g in out:
            lo = max(a["f_lo_khz"], min(x["f_lo_khz"] for x in g))
            hi = min(a["f_hi_khz"], max(x["f_hi_khz"] for x in g))
            narrow = min(a["f_hi_khz"] - a["f_lo_khz"], min(x["f_hi_khz"] - x["f_lo_khz"] for x in g))
            if hi - lo >= min_overlap * max(narrow, 1e-9):
                g.append(a)
                break
        else:
            out.append([a])
    return out


def listen_all(iq, answers, out_dir, rng=None, **kw):
    """Adds a 'listen' entry to every answer worth zooming into; tracks on the
    same frequency share one (over their joined time span)."""
    rng = rng or np.random.default_rng(0)
    nv = extract.noise_variance(np.asarray(iq[:min(len(iq), 32768)]))
    for g in groups(answers):
        # The whole capture at that frequency, as an operator would listen:
        # Morse words and speech pauses split one transmission into tracks.
        span = dict(g[0])
        span.update(track=g[0]["track"], t_start_ms=0.0, t_end_ms=len(iq) / FS * 1e3,
                    f_lo_khz=float(np.median([a["f_lo_khz"] for a in g])),
                    f_hi_khz=float(np.median([a["f_hi_khz"] for a in g])))
        try:
            res = listen_track(iq, span, out_dir, nv, rng, **kw)
        except Exception as e:                   # noqa: BLE001 -- one bad track must not stop the rest
            res = {"kind": "ERROR", "why": f"{type(e).__name__}: {e}"}
        res["tracks"] = [a["track"] for a in g]
        res["t_start_ms"], res["t_end_ms"] = span["t_start_ms"], span["t_end_ms"]
        for a in g:
            a["listen"] = res
            override(a, res)
    return answers


# The zoom looks at the whole track at 50 kS/s; the classifier at 160 us
# windows, too short to hear a voice cycle, so it calls voice radios tones or
# spot jammers (error budget, docs/ACCURACY_PLAN.md P3). On a clear signal the
# zoom's answer wins. Measured on voice-like signals: 70/72 kinds right at
# 30 dB, 65/72 at 20 dB, 62/72 at 15 dB (src/pipeline/analog.py tests).
ZOOM_MIN_SNR_DB = 15.0
ZOOM_HIGH_SNR_DB = 20.0


def zoom_answer(kind, feats):
    """Analog kind -> (answer label, dataset classes it stands for), or None."""
    carrier = feats.get("carrier_frac", 0) >= analog.CARRIER_FRAC
    table = {
        "NBFM": ("NBFM", ["NBFM"]),
        "WBFM": ("WBFM", ["WBFM"]),
        "AM": ("AM_DSB_WC", ["AM_DSB_WC"]),
        "AM_SC": ("AM_DSB_SC", ["AM_DSB_SC"]),
        "USB": ("AM_SSB_WC (USB)" if carrier else "AM_SSB_SC (USB)", ["AM_SSB_WC" if carrier else "AM_SSB_SC"]),
        "LSB": ("AM_SSB_WC (LSB)" if carrier else "AM_SSB_SC (LSB)", ["AM_SSB_WC" if carrier else "AM_SSB_SC"]),
        "CW": ("OOK (Morse CW)", ["OOK"]),
        "CARRIER": ("RADAR_CW / JAM_TONE", ["RADAR_CW", "JAM_TONE"]),
    }
    return table.get(kind)


def override(ans, res):
    """Replace the classifier's answer with the zoom's when the zoom is sure."""
    z = zoom_answer(res.get("kind"), res.get("features") or {})
    snr = (res.get("features") or {}).get("snr_db", -99.0)
    if z is None or snr is None or snr < ZOOM_MIN_SNR_DB:
        return
    label, classes = z
    if label == ans["answer"]:
        return
    ans["classifier_answer"] = ans["answer"]
    ans["answer"], ans["classes"] = label, classes
    ans["band"] = "high" if snr >= ZOOM_HIGH_SNR_DB else "medium"
    ans["decided_by"] = "zoom"
    ans["reason"] = f"zoom-in over the whole track ({snr:.0f} dB): {res.get('why', '')}; " \
                    f"the 160 us classifier said {ans['classifier_answer']}"
