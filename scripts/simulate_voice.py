"""
Simulate noisy radio voice, decode it with Whisper, and score it against the
script: how bad can the audio get before the words are wrong?

    py -3.11 scripts/simulate_voice.py                       # every scenario
    py -3.11 scripts/simulate_voice.py --clips 2 --scenarios clean hiss_+0dB
    py -3.11 scripts/simulate_voice.py --list

Your own clean recording instead of the Windows voices (use a person's real voice):
    py -3.11 scripts/simulate_voice.py --speech me.wav --truth "what I said in it"

Decode a noisy file as it is, no simulation or score:
    py -3.11 scripts/simulate_voice.py --decode recording.wav

Fully offline: the Windows voices are built in, the Whisper model must already be
in ~/.cache/whisper (a missing one stops the run rather than downloading), and
this process's network access is switched off before anything is loaded.

Needs openai-whisper, numpy and scipy (on this laptop: Python 3.11, `py -3.11`).
Writes to --out (results/ is git-ignored): every clip as a WAV to listen to,
results.json and report.txt.

Reading the table:
  WER      word errors / words said. 0% is perfect; over 100% means it invented
           more words than were said.
  flagged  share of the decoded text Whisper itself was unsure of (segment
           avg_logprob < -1.0 or no_speech_prob > 0.6). Compare with WER to see
           whether doubt can be trusted to warn of wrong text.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.intel import voice_sim as vs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOW_LOGPROB, HIGH_NO_SPEECH = -1.0, 0.6


def block_network():
    """Any attempt by this process to open a connection now fails loudly."""
    def refuse(*a, **k):
        raise OSError("network access is switched off in simulate_voice.py")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def load_model(name):
    path = Path.home() / ".cache" / "whisper" / f"{name}.pt"
    if not path.exists():
        sys.exit(f"The Whisper model '{name}' is not on this machine ({path}). This script never downloads. "
                 "While online, run once: py -3.11 -c \"import whisper; whisper.load_model('" + name + "')\"")
    import whisper
    return whisper.load_model(name)


def decode(model, audio, language):
    """Whisper on float32 16 kHz audio -> (text, text with doubtful segments marked, flagged share, segments)."""
    res = model.transcribe(audio, language=language, fp16=False, condition_on_previous_text=False, verbose=None)
    segs, marked, flagged, total = [], [], 0, 0
    for s in res.get("segments", []):
        text = s["text"].strip()
        low = s["avg_logprob"] < LOW_LOGPROB or s["no_speech_prob"] > HIGH_NO_SPEECH
        n = len(vs.normalize(text))
        total += n
        flagged += n if low else 0
        marked.append(f"[? {text}]" if low else text)
        segs.append(dict(start_s=round(s["start"], 2), end_s=round(s["end"], 2), text=text, doubtful=low,
                         avg_logprob=round(s["avg_logprob"], 3), no_speech_prob=round(s["no_speech_prob"], 3)))
    return res.get("text", "").strip(), " ".join(marked), flagged / total if total else 0.0, segs


def clips(args, out):
    """(speech, other talker, script, label) per clip."""
    if args.speech:
        x = vs.read_wav(args.speech)
        if not args.truth:
            sys.exit("--speech needs --truth (the words said in it), otherwise nothing can be scored")
        other = vs.spoken(vs.LINES[3], vs.VOICES[1], out / "speech")
        return [(x, other, args.truth, Path(args.speech).stem)]
    made = []
    for i in range(args.clips):
        line = vs.LINES[i % len(vs.LINES)]
        voice = vs.VOICES[i % len(vs.VOICES)]
        other = vs.spoken(vs.LINES[(i + 3) % len(vs.LINES)], vs.VOICES[(i + 1) % len(vs.VOICES)], out / "speech")
        made.append((vs.spoken(line, voice, out / "speech"), other, line, f"line{i + 1}_{voice.split()[1].lower()}"))
    return made


def main():
    # Noise can make Whisper write any alphabet, which a Windows console cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "results" / "voice_sim" / "run"))
    p.add_argument("--model", default="base", help="Whisper model already in ~/.cache/whisper")
    p.add_argument("--language", default="en", help="en, ms, id ... (empty = detect)")
    p.add_argument("--clips", type=int, default=3, help="how many script lines to use")
    p.add_argument("--scenarios", nargs="+", default=list(vs.SCENARIOS))
    p.add_argument("--speech", help="clean WAV of real speech, used instead of the Windows voices")
    p.add_argument("--truth", help="the words said in --speech")
    p.add_argument("--decode", metavar="WAV", help="just decode this file and print it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="keep the decodes already in --out/results.json")
    p.add_argument("--list", action="store_true", help="list the scenarios and stop")
    args = p.parse_args()
    if args.list:
        print("\n".join(vs.SCENARIOS))
        return
    unknown = set(args.scenarios) - set(vs.SCENARIOS)
    if unknown:
        sys.exit(f"unknown scenarios {sorted(unknown)}; see --list")

    language = args.language or None
    if args.decode:
        block_network()
        model = load_model(args.model)
        _, marked, flagged, segs = decode(model, vs.read_wav(args.decode), language)
        for s in segs:
            print(f"{'?' if s['doubtful'] else ' '} [{s['start_s']:7.1f} - {s['end_s']:7.1f}] {s['text']}")
        print(f"\n{flagged:.0%} of the words are from doubtful segments (marked ?)")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = clips(args, out)                     # the Windows voices run in their own local process
    block_network()
    model = load_model(args.model)
    rows = []
    if args.resume and (out / "results.json").exists():
        rows = json.loads((out / "results.json").read_text(encoding="utf-8"))["rows"]
        print(f"resuming: {len(rows)} decodes already done", flush=True)
    done = {(r["scenario"], r["clip"]) for r in rows}
    t0 = time.time()
    for name in args.scenarios:
        for k, (speech, other, script, label) in enumerate(data):
            if (name, label) in done:
                continue
            # The seed is per (scenario, clip), so re-running one scenario gives the same audio.
            rng = np.random.default_rng([args.seed, list(vs.SCENARIOS).index(name), k])
            noisy = vs.simulate(name, speech, other, rng)
            vs.save_wav(out / "wav" / f"{name}__{label}.wav", noisy)
            t1 = time.time()
            text, marked, flagged, segs = decode(model, noisy, language)
            truth = "" if name in vs.NO_SPEECH else script
            score = vs.wer(truth, text)
            rows.append(dict(scenario=name, clip=label, truth=truth, text=text, marked=marked, flagged=flagged,
                             seconds_audio=round(len(noisy) / vs.FS, 1), seconds_decode=round(time.time() - t1, 1),
                             segments=segs, **score))
            print(f"{name:>18} {label:<18} WER {score['wer']:>5.0%}  {text[:70]}", flush=True)
            (out / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1), encoding="utf-8")
    report(rows, args, out, time.time() - t0)


def report(rows, args, out, seconds):
    lines = [f"model {args.model}, language {args.language or 'detect'}, {len(rows)} decodes in {seconds:.0f} s, "
             f"speed {sum(r['seconds_audio'] for r in rows) / max(sum(r['seconds_decode'] for r in rows), 1e-9):.1f}x real time",
             "", f"{'scenario':>18} {'WER':>6} {'sub':>5} {'del':>5} {'ins':>5} {'words said':>11} {'flagged':>8}"]
    for name in args.scenarios:
        r = [x for x in rows if x["scenario"] == name]
        words = sum(x["truth_words"] for x in r)
        errors = sum(x["sub"] + x["del"] + x["ins"] for x in r)
        wer = f"{errors / words:.0%}" if words else f"{sum(x['hyp_words'] for x in r)} w"
        lines.append(f"{name:>18} {wer:>6} {sum(x['sub'] for x in r):>5} {sum(x['del'] for x in r):>5} "
                     f"{sum(x['ins'] for x in r):>5} {words:>11} {np.mean([x['flagged'] for x in r]):>8.0%}")
    lines += ["", "(noise_only has no speech: its figure is how many words were invented)", "", "Examples, first clip of each:"]
    for name in args.scenarios:
        r = next(x for x in rows if x["scenario"] == name)
        lines += [f"  {name}", f"    said:    {r['truth'] or '(nothing)'}", f"    decoded: {r['marked'] or '(nothing)'}"]
    text = "\n".join(lines)
    print("\n" + text)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nWrote {out / 'report.txt'}, results.json and the clips in {out / 'wav'}")


if __name__ == "__main__":
    main()
