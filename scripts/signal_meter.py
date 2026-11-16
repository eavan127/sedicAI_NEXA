r"""Live signal meter for setting the RSPduo gain before a walkie-talkie test.

Reads SDRconnect's IQ stream (WebSocket API, port 5454) and prints, once a
second, how strong and how wide the strongest signal is. Hold the
walkie-talkie's talk button and turn SDRconnect's RF Gain / IF Gain down until
it says GOOD: an overloaded walkie-talkie looks 60-70 dB strong and 50-170 kHz
wide, and NEXA then mistakes it for single sideband and cannot read the words.

    .\.venv\Scripts\python.exe scripts/signal_meter.py

Needs SDRconnect open, the radio playing (green play pressed) and
Preferences -> WebSocket Server on. Stop with Ctrl+C.
"""
import collections
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.live_sdrconnect import parse_binary  # noqa: E402

NFFT = 4096
GOOD_DB = (15, 45)        # strong enough to read, not so strong that it distorts
GOOD_WIDTH_KHZ = 20       # a walkie-talkie channel is about 12.5 kHz


def main():
    buf, info = collections.deque(maxlen=200), {}
    ws = connect("ws://127.0.0.1:5454", max_size=None, open_timeout=10)

    def rx():
        for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                iq = parse_binary(msg)
                if iq is not None:
                    buf.append(iq)
            else:
                j = json.loads(msg)
                if j.get("property"):
                    info[j["property"]] = j.get("value")

    threading.Thread(target=rx, daemon=True).start()
    ws.send(json.dumps({"event_type": "get_property", "property": "device_sample_rate", "value": ""}))
    ws.send(json.dumps({"event_type": "get_property", "property": "device_center_frequency", "value": ""}))
    ws.send(json.dumps({"event_type": "iq_stream_enable", "property": "", "value": "true"}))
    print("Hold the walkie-talkie's talk button and watch the verdict. Ctrl+C to stop.\n")
    window = np.hanning(NFFT)
    while True:
        time.sleep(1.0)
        chunks = [buf.popleft() for _ in range(len(buf))]
        if not chunks:
            print("  no IQ: press play in SDRconnect")
            continue
        fs = float(info.get("device_sample_rate") or 2e6)
        centre = float(info.get("device_center_frequency") or 0)
        x = np.concatenate(chunks)
        k = len(x) // NFFT
        if k < 4:
            continue
        p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(x[:k * NFFT].reshape(k, NFFT) * window, axis=1), axes=1)) ** 2, axis=0)
        db = 10 * np.log10(p + 1e-20)
        f = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
        db[np.abs(f) < 10e3] = np.median(db)                      # the RSPduo's own centre spike
        noise = np.median(db)
        j = int(np.argmax(db))
        above = db[j] - noise
        # width: the stretch around the peak that stays within 30 dB of it and clearly above the noise
        level = max(db[j] - 30, noise + 6)
        lo = hi = j
        while lo > 0 and db[lo - 1] > level:
            lo -= 1
        while hi < NFFT - 1 and db[hi + 1] > level:
            hi += 1
        width_khz = (f[hi] - f[lo]) / 1e3 + fs / NFFT / 1e3
        where = f"{(centre + f[j]) / 1e6:.4f} MHz" if centre else f"{f[j] / 1e3:+.1f} kHz"
        if above < 10:
            verdict = "nothing on air (press the talk button)"
        elif above > GOOD_DB[1] or width_khz > GOOD_WIDTH_KHZ:
            verdict = "TOO STRONG: turn RF Gain left / IF Gain down, or move the walkie-talkie away"
        elif above < GOOD_DB[0]:
            verdict = "a bit weak: move closer or turn gain up a little"
        else:
            verdict = "GOOD: NEXA can read this"
        print(f"  {where}   {above:5.1f} dB above noise   {width_khz:6.1f} kHz wide   ->  {verdict}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
