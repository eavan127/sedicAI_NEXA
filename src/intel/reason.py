"""Local-LLM reading of decoded traffic, through Ollama (http://localhost:11434).

The model gets the decoded text plus facts measured in Python (call-sign-like
tokens, Q-codes, five-character groups, speed) and returns JSON: a short
summary, a category, call signs, places, key points and EVIDENCE quotes.

Small local models invent things, so nothing is taken on trust:
  * call signs, places and evidence quotes that do not literally occur in the
    decoded text are removed (and counted in `removed_ungrounded`);
  * the category must be one of CATEGORIES;
  * confidence drops to "low" when the decode is poor or evidence was removed.
The dashboard shows the decoded text next to the reading, so a person can
always check the model against the source.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
CATEGORIES = ["calling / contact", "message traffic", "coded groups", "beacon / repeated",
              "weather", "position / navigation", "test / tuning", "unreadable", "other"]
Q_CODES = {"QRA", "QRG", "QRK", "QRL", "QRM", "QRN", "QRO", "QRP", "QRQ", "QRS", "QRT", "QRU", "QRV",
           "QRX", "QRZ", "QSB", "QSL", "QSO", "QSP", "QSY", "QTC", "QTH", "QTR", "QSA", "QSK"}
CALL_RE = re.compile(r"\b(?:[A-Z]{1,2}|[0-9][A-Z])[0-9][A-Z]{1,4}\b")
GROUP_RE = re.compile(r"\b[A-Z0-9]{5}\b")
MAX_CHARS = 6000


def facts(text, segments=None):
    """What can be counted without a model."""
    words = text.split()
    calls = Counter(m for m in CALL_RE.findall(text) if any(c.isalpha() for c in m))
    groups = [w for w in words if GROUP_RE.fullmatch(w) and not CALL_RE.fullmatch(w)]
    out = {
        "call_sign_like": calls.most_common(15),
        "q_codes": Counter(w for w in words if w in Q_CODES).most_common(10),
        "cq_calls": words.count("CQ"),
        "five_char_groups_share": round(len(groups) / max(len(words), 1), 3),
        "top_words": Counter(w for w in words if len(w) > 1 and "*" not in w).most_common(15),
        "unreadable_share": round(text.count("*") / max(len(text.replace(" ", "")), 1), 3),
    }
    if segments:
        wpm = [s["wpm"] for s in segments if "wpm" in s]
        if wpm:
            out["wpm_range"] = [min(wpm), max(wpm)]
    return out


def digest(segments, kind="morse", limit=MAX_CHARS):
    """Readable transmissions spread across the recording, with times, under `limit` chars."""
    def stamp(sec):
        sec = int(sec)
        return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    def cut(t, n=400):                                  # a 20-minute transmission must not fill the prompt alone
        return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"
    lines = [f"[{stamp(s.get('start_s', s.get('clip_start_s', 0)))}] {cut(s['text'])}"
             for s in segments if s.get("unreadable", 0) < 0.3 and s["text"].strip()]
    if sum(len(x) + 1 for x in lines) <= limit:
        return "\n".join(lines), len(lines), len(lines)
    budget = max(1, limit // max(1, int(sum(len(x) for x in lines) / len(lines)) + 1))
    idx = sorted({int(round(i)) for i in [k * (len(lines) - 1) / max(budget - 1, 1) for k in range(budget)]})
    picked = [lines[i] for i in idx]
    while picked and sum(len(x) + 1 for x in picked) > limit:
        picked.pop()
    return "\n".join(picked), len(picked), len(lines)


PROMPT = """You are helping a radio analyst read {what} decoded automatically from a receiver recording.
The decode can contain errors ('*' = unreadable letter). Use ONLY what is in the text below.
Never invent call signs, places, names or meaning that are not in the text. If the text is too
garbled to say anything, use category "unreadable" and say so.

Recording: {name}, {duration}, {active} minutes with signal.
Measured in Python (exact counts): {facts}

Decoded text ({n_shown} of {n_total} transmissions, with time from the start of the recording):
---
{text}
---

Answer with JSON only, exactly these keys:
{{"summary": "2-3 plain sentences on what this traffic is",
  "category": one of {categories},
  "language": "main language or 'abbreviations only'",
  "translation_en": "English meaning of any non-English plain text, else empty string",
  "call_signs": ["only tokens copied exactly from the text"],
  "places": ["only place names copied exactly from the text"],
  "key_points": ["up to 5 short points"],
  "evidence": [{{"quote": "a short piece copied EXACTLY from the text", "why": "what it shows"}}],
  "confidence": "low" | "medium" | "high",
  "caveats": "what could be wrong"}}"""


def ask(prompt, model=DEFAULT_MODEL, url=DEFAULT_URL, timeout=600):
    body = json.dumps({"model": model, "stream": False, "format": "json", "keep_alive": "30m",
                       "options": {"temperature": 0.1, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def _norm(s):
    return " ".join(str(s).upper().split())


def ground(reading, source):
    """Keep only what the source text supports; see the module docstring."""
    src = _norm(source)
    removed = 0
    out = {k: reading.get(k) for k in ("summary", "language", "translation_en", "caveats")}
    out = {k: (v if isinstance(v, str) else "") for k, v in out.items()}
    cat = str(reading.get("category", "other")).strip().lower()
    out["category"] = next((c for c in CATEGORIES if c == cat or c.split(" /")[0] == cat), "other")
    for key in ("call_signs", "places"):
        vals = [v for v in reading.get(key) or [] if isinstance(v, str) and v.strip()]
        keep = [v for v in vals if _norm(v) in src]
        if key == "call_signs":                        # VVV, QTC or 48213 are not call signs
            keep = [v for v in keep if CALL_RE.fullmatch(_norm(v))]
        removed += len(vals) - len(keep)
        out[key] = list(dict.fromkeys(keep))
    pts = reading.get("key_points") or []
    out["key_points"] = [p for p in pts if isinstance(p, str)][:5]
    ev = []
    for e in reading.get("evidence") or []:
        if isinstance(e, dict) and isinstance(e.get("quote"), str) and _norm(e["quote"]) and _norm(e["quote"]) in src:
            ev.append({"quote": e["quote"].strip(), "why": str(e.get("why", ""))})
        else:
            removed += 1
    out["evidence"] = ev[:6]
    conf = str(reading.get("confidence", "low")).lower()
    out["confidence"] = conf if conf in ("low", "medium", "high") else "low"
    if not ev:
        out["confidence"] = "low"                      # nothing checkable backs the reading
    elif removed and out["confidence"] == "high":
        out["confidence"] = "medium"                   # the model made at least one thing up
    out["removed_ungrounded"] = removed
    return out


def read_recording(name, segments, kind="morse", duration="", active_min=0, model=DEFAULT_MODEL,
                   url=DEFAULT_URL):
    """segments -> grounded reading, or {"error": ...} if Ollama is not reachable or answers badly."""
    text, n_shown, n_total = digest(segments, kind)
    full = "\n".join(s["text"] for s in segments)
    f = facts(full, segments)
    if not text.strip():
        return {"category": "unreadable", "summary": "Nothing readable was decoded.", "confidence": "high",
                "call_signs": [], "places": [], "key_points": [], "evidence": [], "language": "",
                "translation_en": "", "caveats": "", "removed_ungrounded": 0, "facts": f, "model": None}
    prompt = PROMPT.format(what="Morse (CW) traffic" if kind == "morse" else "speech transcribed by Whisper",
                           name=name, duration=duration, active=active_min, facts=json.dumps(f),
                           n_shown=n_shown, n_total=n_total, text=text, categories=json.dumps(CATEGORIES))
    try:
        raw = ask(prompt, model, url)
        reading = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": f"Ollama not reachable at {url}: {e}", "facts": f}
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"model answer was not JSON: {e}", "facts": f}
    out = ground(reading, text)
    out.update({"facts": f, "model": model, "shown_transmissions": n_shown, "total_transmissions": n_total})
    return out
