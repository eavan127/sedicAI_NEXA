"""Comms intel: turn recorded receiver AUDIO into text and a short local-LLM
reading of it, for the dashboard's Comms Intel page.

    audio.py    read long PCM WAVs in chunks (8 kHz / 8-bit recorder files are hours long)
    morse.py    find the CW tone, key-down/up timing, dots and dashes -> text
    voice.py    Whisper speech-to-text for recordings that are voice, not Morse
    reason.py   Ollama (local) summary / category / evidence, checked against the text

Everything runs on this machine: no audio, text or result leaves it.
scripts/comms_intel.py runs the whole chain over a folder.
"""
