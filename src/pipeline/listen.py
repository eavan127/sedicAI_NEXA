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
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly

from src.config import CLASS_GROUPS, CFG
from src.intel import audio as wavio
from src.pipeline import analog, demod, extract

FS = float(CFG["signal"]["fs"])
# The whole ASK group, not only OOK: Morse is OOK, and the classifier cannot
# tell ASK sizes apart (resolve answers "4ASK / 8ASK"). SED-44: every word of
# a 20 WPM Morse recording came out "4ASK / 8ASK" and was never decoded. The
# zoom decides what it really is; a real ASK data signal is named UNKNOWN or
# kept as the classifier said.
LISTEN_CLASSES = set(CLASS_GROUPS["ANALOG"]) | set(CLASS_GROUPS["ASK"]) | {"RADAR_CW", "JAM_TONE", "JAM_SPOT"}
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


MIX_BLOCK = 1 << 20


def zoomed_cut(iq, t0, t1, f_lo, f_hi, noise_var, rng, decim=WIDE_DECIM, fs=FS):
    """analog.zoom(extract.cut(...).filled, decim), zoomed FIRST: the box mixed
    to 0 Hz and decimated, then everything outside it (plus extract's margin)
    replaced by noise at the floor. The same signal for 1/decim of the memory:
    the full-rate cut of a 10 s recording (34 M samples, ~6 double-precision
    copies) ran out of memory on the SED-44 Morse files."""
    n = len(iq)
    t0, t1 = int(max(0, t0)), int(min(n, t1))
    if t1 - t0 < extract.WINDOW_LEN:
        mid = (t0 + t1) // 2
        t0 = max(0, min(mid - extract.WINDOW_LEN // 2, n - extract.WINDOW_LEN))
        t1 = min(n, t0 + extract.WINDOW_LEN)
    fc = (f_lo + f_hi) / 2
    half = (f_hi - f_lo) / 2 + max(extract.MIN_MARGIN_HZ, extract.CUT_MARGIN_FRAC * (f_hi - f_lo))
    x = np.empty(t1 - t0, np.complex64)
    for s in range(t0, t1, MIX_BLOCK):              # mix to 0 Hz block by block: no full-length phase array
        e = min(t1, s + MIX_BLOCK)
        x[s - t0:e - t0] = iq[s:e] * np.exp(-2j * np.pi * fc * np.arange(s, e) / fs).astype(np.complex64)
    # to FS / decim (400 kS/s) from whatever rate the recording has: decimate a 3.2 MS/s
    # capture, or interpolate a 250 kS/s file the zoom-in listens to at its own rate
    fa = FS / decim
    r = Fraction(fa / fs).limit_denominator(1000)
    y = resample_poly(x, r.numerator, r.denominator)
    del x
    m = len(y)
    kc = half * m / fa                                # bins with |f| <= half are the box
    a, b = int(np.floor(kc)) + 1, int(np.ceil(m - kc))   # Y[a:b] lies outside it
    if b > a:
        Y = sfft.fft(y, overwrite_x=True)
        # white noise of variance noise_var at fs has density noise_var / fs: at the zoom's
        # rate fa that is noise_var * fa / fs per sample (noise_var / decim at fs = FS)
        scale = np.float32(np.sqrt(m * noise_var * fa / fs / 2))
        Y[a:b] = (rng.standard_normal(b - a, dtype=np.float32) * scale
                  + 1j * (rng.standard_normal(b - a, dtype=np.float32) * scale))
        y = sfft.ifft(Y, overwrite_x=True)
    return np.asarray(y, dtype=np.complex64)


def listen_track(iq, ans, out_dir, noise_var, rng, whisper_model=None, language=None, llm_model=None,
                 use_whisper=True, use_llm=True, fs=FS):
    t0 = int(ans["t_start_ms"] * fs / 1e3)
    t1 = int(ans["t_end_ms"] * fs / 1e3)
    f_lo, f_hi = ans["f_lo_khz"] * 1e3, ans["f_hi_khz"] * 1e3
    # Two-step zoom: 400 kS/s first, find the signal itself inside the (often
    # much wider) box, then centre on it and zoom to 50 kS/s for voice / Morse.
    fa1 = FS / WIDE_DECIM
    y1 = zoomed_cut(iq, t0, t1, f_lo, f_hi, noise_var, rng, fs=fs)   # noise-filled outside the box: the floor stays measurable
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
    demod_kind, slow = kind, None
    if not wide:                                    # second opinion from the slow-path model (src/pipeline/narrowband.py)
        from src.pipeline import narrowband
        kind, why, demod_kind, slow = narrowband.refine(kind, why, y, fa)
    dig = None
    if not wide:                                    # RTTY / PSK31 text (src/intel/digital.py)
        from src.intel import digital
        was = kind
        kind, dig = digital.read(kind, y, fa, slow)
        if kind != was:
            why, demod_kind = dig["why"], "USB"
    out = {"kind": kind, "why": why, "duration_s": round(len(y) / fa, 3),
           "features": {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                        for k, v in feats.items() if k != "band"}}
    if slow:
        out["slow_path"] = slow
    if kind == "UNKNOWN":
        return out
    sound = demod.demodulate(y, fa, demod_kind, feats)
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
        if dig is not None:
            from src.intel import digital
            out.update(digital.listen_fields(dig))
            segments, text_kind = dig["segments"], "digital"
        elif kind == "CW":
            from src.intel import morse
            m = morse.analyse(info)
            segments = m["segments"]
            out["morse"] = " / ".join(s["text"] for s in segments)
            out["morse_detail"] = m
            text_kind = "morse"
        elif kind in VOICE_KINDS and use_whisper and out["duration_s"] >= MIN_SPEECH_S:
            from src.intel import voice
            tr = voice.transcribe(info, model=whisper_model or voice.DEFAULT_MODEL, max_minutes=1.0, language=language)
            out["transcript"] = tr.get("text", "")
            out["language"] = tr.get("language")
            out["whisper_model"] = tr.get("model")
            out["doubtful_share"] = tr.get("doubtful_share", 0.0)
            out["segments"] = [{"start_s": s["clip_start_s"], "end_s": s["clip_end_s"], "text": s["text"],
                                "doubtful": s["doubtful"]} for s in tr.get("segments", [])]
            # Every segment, for the Comms Intel store (src/intel/store.py reads speech_segments).
            out["speech_segments"] = [{"start_s": s["start_s"], "end_s": s["end_s"], "text": s["text"]}
                                      for s in out["segments"]]
            # The AI reading only gets what Whisper was sure of, so it cannot build
            # a confident summary on text Whisper itself doubted.
            segments = [{"start_s": s["start_s"], "text": s["text"]} for s in out["segments"] if not s["doubtful"]]
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


def listen_all(iq, answers, out_dir, rng=None, fs=FS, **kw):
    """Adds a 'listen' entry to every answer worth zooming into; tracks on the
    same frequency share one (over their joined time span). iq may be the
    recording at its own rate fs (and longer than the part the detector saw):
    the zoom-in then hears the whole transmission, e.g. a 26 s Morse message
    of which only the first seconds were scanned at 3.2 MS/s."""
    rng = rng or np.random.default_rng(0)
    nv = extract.noise_variance(np.asarray(iq[:min(len(iq), 32768)]))
    for g in groups(answers):
        # The whole capture at that frequency, as an operator would listen:
        # Morse words and speech pauses split one transmission into tracks.
        span = dict(g[0])
        span.update(track=g[0]["track"], t_start_ms=0.0, t_end_ms=len(iq) / fs * 1e3,
                    f_lo_khz=float(np.median([a["f_lo_khz"] for a in g])),
                    f_hi_khz=float(np.median([a["f_hi_khz"] for a in g])))
        try:
            res = listen_track(iq, span, out_dir, nv, rng, fs=fs, **kw)
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
# Kinds named by a carrier, which stays easy to see in noise: AM ("strong carrier with
# sidebands on both sides") and Morse (one carrier keyed on and off). Their zoom answer
# wins from 10 dB (band "medium" below 20 dB). Measured 13 Nov 2026 on the demo files:
# a Malay AM clip at 14.6 dB stayed "4ASK / 8ASK uncertain" under the 15 dB rule.
ZOOM_MIN_SNR_DB_KIND = {"AM": 10.0, "CW": 10.0}


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
        # Named by the slow-path model (src/pipeline/narrowband.py): slow HF data modes.
        "RTTY": ("2FSK (RTTY)", ["2FSK"]),
        "PSK31": ("BPSK (PSK31-type)", ["BPSK"]),
        "AFSK": ("AFSK", ["AFSK"]),
    }
    return table.get(kind)


def override(ans, res):
    """Replace the classifier's answer with the zoom's when the zoom is sure."""
    z = zoom_answer(res.get("kind"), res.get("features") or {})
    snr = (res.get("features") or {}).get("snr_db", -99.0)
    if z is None or snr is None or snr < ZOOM_MIN_SNR_DB_KIND.get(res.get("kind"), ZOOM_MIN_SNR_DB):
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
