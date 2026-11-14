"""The dashboard Assistant's local-AI answers (Ollama on this machine).

The page's own keyword matcher (web/chatbot.js) still answers the structured
questions (list, compare, count, "which uploads had X"). Everything else comes
here: build_context() gathers NEXA's real records (captures, alerts, review
queue, decoded intercepts, the current model's measured results) into a short
plain-text sheet, and answer() asks the local model to answer ONLY from that
sheet. Nothing leaves the machine.

Small local models still make things up, so the prompt says "I don't have that
in the records" is the right answer when the sheet does not contain it, and the
page labels the reply as the AI's, next to the records it was given.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import reason

MAX_QUESTION = 500

SYSTEM_FACTS = """How NEXA works (fixed facts):
- Everything runs on this laptop; no data goes to the internet.
- Inputs: SDRconnect (the RSPduo radio, live or playing an IQ file), the built-in simulated receiver (test scenarios, overlapping signals), and KiwiSDR (online short-wave receivers).
- Chain: a YOLO detector boxes each transmitter on the waterfall -> a 60-class model names it (radars, jammers, frequency hopping, digital and analog modulations) and votes over the whole transmission -> for voice or Morse, NEXA zooms in, demodulates (NBFM / AM / SSB / CW), decodes Morse or turns speech into text with Whisper, and the local AI (Ollama) writes a reading and can translate to English.
- Verdict tiers: Hostile (jammers), Military (radars, hopping), Civilian (ordinary modulations), Empty (nothing found).
- Human in the loop: operators correct wrong labels; an analyst approves corrections. Retraining uses approved corrections, is started by a person, and a new model is only offered if it passes a fixed test. Every model version is kept. Every action is in the audit trail, which cannot be edited.
- Review queue: Alerts (military or hostile emitters), Corrections (waiting for an analyst), Low confidence (close calls a person should check)."""

PROMPT = """You are the assistant inside NEXA, a radio signal intelligence system. Answer the
user's question using ONLY the facts and records below. Be clear and short (1-4 sentences,
plain English, no jargon without a short explanation). Write natural sentences for a
person: never paste a record line as it is (say "a voice transmission at 3:14 pm on 14 Nov",
not "intercept 2026-11-14T15:14 voice"). A count of 0 is a real answer: say so plainly
(e.g. "Nothing is waiting for review: 0 alerts, 0 corrections and 0 low-confidence detections").
The "answer" must never be empty. Quote numbers, times, file names and
call signs exactly as written. If the answer is not in the records, say "I don't have that in
NEXA's records" and suggest which page would show it. Never invent detections, numbers or
meanings.

{facts}

RECORDS (read from this laptop's database just now):
{records}

QUESTION: {question}

Reply with the answer only, in plain text (no JSON, no heading). If one page of the dashboard
shows it, end with one line: "See: <page>" using one of: SDR Receiver, KiwiSDR, Intercepts,
Alerts, Corrections, Low Confidence, Captures, Audit Trail, Reports, Performance, Model."""


def _when(iso):
    """'2026-11-14T15:14:45' -> 'on 14 Nov 2026 at 15:14'."""
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return ""
    return f"on {d.day} {d:%b %Y} at {d:%H:%M}"


def _pct(v):
    return f"{round(v * 100)}%" if isinstance(v, (int, float)) else "?"


def build_context(db, intel_dir: Path, web_data: Path, n_recent: int = 8) -> str:
    """NEXA's records as a compact plain-text sheet for the model."""
    lines = []
    caps = db.list_analyses(limit=500)
    found = [c for c in caps if (c.get("classes_detected") or [])]
    verdicts = Counter(c.get("verdict") or "?" for c in caps)
    classes = Counter(k for c in caps for k in (c.get("classes_detected") or []))
    lines.append(f"Captures stored: {len(caps)} ({', '.join(f'{v} {n}' for v, n in verdicts.most_common())}).")
    if classes:
        lines.append("Classes seen most (captures): " + ", ".join(f"{k} {n}" for k, n in classes.most_common(10)) + ".")
    for c in found[:n_recent]:
        snr = c.get("snr_db")
        lines.append(f"- A capture {_when(c.get('created_at'))} ('{c.get('file_name') or c.get('case_note') or c.get('source')}'): "
                     f"verdict {c.get('verdict')}, signals found {', '.join(c.get('classes_detected') or [])}"
                     f"{f', SNR {snr:.1f} dB' if isinstance(snr, (int, float)) else ''}"
                     f"{', reviewed by ' + c['reviewed_by'] if c.get('reviewed_by') else ''}.")
    alerts = [c for c in caps if c.get("alert_level") and not c.get("reviewed_by")]
    low = [c for c in caps if c.get("needs_review") and not c.get("reviewed_by")]
    stats = db.correction_stats()
    lines.append(f"Review queue: {len(alerts)} alerts waiting, {stats.get('pending', 0)} corrections waiting for an analyst, "
                 f"{len(low)} low-confidence detections to check. Corrections approved so far: {stats.get('approved', 0)}.")
    active = db.active_model("single")
    if active:
        lines.append(f"Model in use: version {active.get('version')} (retrained from corrections).")

    index = Path(intel_dir) / "index.json"
    if index.is_file():
        try:
            recs = [r for r in json.loads(index.read_text(encoding="utf-8")).get("recordings", []) if r.get("kind") != "silent"]
        except (OSError, json.JSONDecodeError):
            recs = []
        kinds = Counter(r.get("kind") for r in recs)
        lines.append(f"Decoded intercepts: {len(recs)} ({', '.join(f'{k} {n}' for k, n in kinds.most_common())}).")
        kind_word = {"voice": "voice", "morse": "Morse code"}
        for r in list(reversed(recs))[:n_recent]:
            calls = ", ".join((r.get("call_signs") or [])[:4])
            name = str(r.get("file") or "").split(" · ")[0].replace("input_", "")
            src = "KiwiSDR" if "kiwisdr" in name.lower() else "SDRconnect live" if name.upper().startswith("LIVE") else f"file {name}"
            lines.append(f"- A {kind_word.get(r.get('kind'), r.get('kind'))} transmission recorded {_when(r.get('recorded_at'))}"
                         f" (source: {src}){f', call signs {calls}' if calls else ''}. What it says: {str(r.get('summary') or 'not read yet')[:220]}")

    cm = Path(web_data) / "current_model.json"
    if cm.is_file():
        try:
            m = json.loads(cm.read_text(encoding="utf-8"))
            fam = "; ".join(f"{f.get('family')} {_pct(f.get('right'))}" for f in m.get("families", []))
            lines.append(f"The model names {m.get('n_classes')} signal types. Measured accuracy on the test set: "
                         f"{_pct(m.get('top1'))} of single 0.16 ms windows named right across all signal strengths; "
                         f"{_pct(m.get('top1_clear'))} of single windows right on clear signals (10 dB and above); "
                         f"{_pct(m.get('vote_clear'))} right on clear signals when it votes over a whole transmission "
                         f"(this is how it is used in practice). Accuracy by family (single windows): {fam}.")
        except (OSError, json.JSONDecodeError):
            pass
    return "\n".join(lines)


def answer(question: str, context: str, model=None, url=reason.DEFAULT_URL, timeout=120) -> dict:
    """-> {"answer", "page"}; raises when Ollama cannot be reached or answers badly.
    The model is NEXA_ASSISTANT_MODEL when set (e.g. llama3.1:8b on a laptop with 16 GB+),
    else the system default (llama3.2:3b)."""
    model = model or os.environ.get("NEXA_ASSISTANT_MODEL") or reason.DEFAULT_MODEL
    q = " ".join(str(question or "").split())[:MAX_QUESTION]
    if not q:
        raise ValueError("empty question")
    body = json.dumps({"model": model, "stream": False, "keep_alive": "30m",
                       "options": {"temperature": 0.1, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": PROMPT.format(
                           facts=SYSTEM_FACTS, records=context or "(no records yet)", question=q)}]}).encode()
    req = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["message"]["content"].strip()
    page = ""
    m = re.search(r"\s*See:\s*(.+?)\s*$", text)
    if m:
        page, text = m.group(1).strip(" .*"), text[:m.start()].strip()
    if not text:
        raise ValueError("the model gave no answer")
    return {"answer": text, "page": page}
