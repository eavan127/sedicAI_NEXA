"""Find a walkie-talkie's frequency with the RSPduo, through SDRconnect's WebSocket API.

Pass 1 (quiet): steps across the band and records what is always there.
Pass 2 and 3 (talk): the same steps while you hold the talk button.
Whatever rose clearly above the quiet pass is your walkie-talkie.
At the end SDRconnect is tuned to it (NFM, 12.5 kHz), and the answer is saved
to results/walkie_found.json.

Needs SDRconnect open, the RSPduo playing (green play pressed) and
Preferences -> WebSocket Server on. Press ENTER in the terminal before each
pass, as it asks.

    .\.venv\Scripts\python.exe scripts/find_walkie.py          # 400-480 MHz (UHF walkie-talkies)
    .\.venv\Scripts\python.exe scripts/find_walkie.py --vhf    # 136-174 MHz too
"""
import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.live_sdrconnect import parse_binary  # noqa: E402

NFFT = 4096
SETTLE_S = 0.35
STEP_HZ = 1.5e6
HALF_SPAN_HZ = 0.8e6       # use only the middle of each 2 MHz view
DC_HZ = 10e3               # the RSPduo's own spike at the centre
PMR446 = [446.00625e6 + 12.5e3 * i for i in range(16)]


def cmd(ev, value="", prop=""):
    return json.dumps({"event_type": ev, "property": prop, "value": str(value)})


class Radio:
    def __init__(self, url):
        self.ws = connect(url, max_size=None, open_timeout=10)
        self.buf = collections.deque(maxlen=64)
        self.info = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._rx, daemon=True).start()
        for p in ("device_sample_rate", "device_center_frequency", "started"):
            self.ws.send(cmd("get_property", prop=p))
        self.ws.send(cmd("iq_stream_enable", "true"))

    def _rx(self):
        for msg in self.ws:
            if isinstance(msg, (bytes, bytearray)):
                iq = parse_binary(msg)
                if iq is not None:
                    with self.lock:
                        self.buf.append(iq)
            else:
                try:
                    j = json.loads(msg)
                except ValueError:
                    continue
                if j.get("property"):
                    self.info[j["property"]] = j.get("value")

    def tune(self, f):
        self.ws.send(cmd("set_property", int(f), "device_center_frequency"))
        self.ws.send(cmd("set_property", int(f), "device_vfo_frequency"))

    def grab(self, n=NFFT * 32, timeout=3.0):
        with self.lock:
            self.buf.clear()
        t0, got = time.time(), []
        while sum(len(x) for x in got) < n and time.time() - t0 < timeout:
            with self.lock:
                while self.buf:
                    got.append(self.buf.popleft())
            time.sleep(0.01)
        return np.concatenate(got)[:n] if got else None


def psd_db(iq):
    k = len(iq) // NFFT
    x = iq[:k * NFFT].reshape(k, NFFT) * np.hanning(NFFT)
    p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2, axis=0)
    return 10 * np.log10(p + 1e-20)


def sweep(radio, centres, fs, label):
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
    keep = (np.abs(freqs) <= HALF_SPAN_HZ) & (np.abs(freqs) >= DC_HZ)
    out = []
    for i, c in enumerate(centres):
        radio.tune(c)
        time.sleep(SETTLE_S)
        iq = radio.grab()
        if iq is None or len(iq) < NFFT * 4:
            print(f"\n  no IQ at {c/1e6:.2f} MHz -- is the radio started (green play) in SDRconnect?")
            out.append(None)
            continue
        out.append(psd_db(iq)[keep])
        bar = "#" * int(30 * (i + 1) / len(centres))
        print(f"\r  {label}: {c/1e6:7.2f} MHz  [{bar:<30}]", end="", flush=True)
    print()
    return out, freqs[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--vhf", action="store_true", help="also scan 136-174 MHz")
    ap.add_argument("--talk-passes", type=int, default=2)
    a = ap.parse_args()

    radio = Radio(f"ws://{a.host}:5454")
    print("Connecting to SDRconnect ...")
    if radio.grab(NFFT * 4, timeout=6) is None:
        sys.exit("No IQ from SDRconnect. Press the green PLAY button in SDRconnect, then run this again.")
    fs = float(radio.info.get("device_sample_rate") or 2e6)
    print(f"Connected. Sample rate {fs/1e6:.1f} MS/s.\n")

    centres = list(np.arange(400.35e6, 480e6, STEP_HZ))
    if a.vhf:
        centres = list(np.arange(136.35e6, 174e6, STEP_HZ)) + centres
    secs = len(centres) * (SETTLE_S + 0.1)

    print("=" * 64)
    print(f" PASS 1 of {1 + a.talk_passes}: QUIET.  Do NOT press the talk button.  (~{secs:.0f} s)")
    print("=" * 64)
    input("  Walkie-talkie ON, talk button NOT pressed. Press ENTER here to start ... ")
    quiet, rel = sweep(radio, centres, fs, "quiet")

    hits = {}
    for p in range(a.talk_passes):
        print()
        print("=" * 64)
        print(f" PASS {p + 2}: HOLD THE TALK BUTTON NOW and keep talking or counting")
        print(f" (walkie-talkie 2-3 m from the antenna).  ~{secs:.0f} s")
        print("=" * 64)
        input("  Press the walkie-talkie's TALK button and hold it, then press ENTER here ... ")
        talk, _ = sweep(radio, centres, fs, "TALK ")
        print("  You can let go of the talk button now.")
        for c, q, t in zip(centres, quiet, talk):
            if q is None or t is None:
                continue
            d = t - q
            j = int(np.argmax(d))
            f = c + rel[j]
            key = round(f / 25e3)                          # same channel across passes
            if d[j] > hits.get(key, (0, 0, 0))[1]:
                hits[key] = (f, float(d[j]), float(t[j]))

    best = sorted(hits.values(), key=lambda h: -h[1])[:5]
    print("\nStrongest rises while you talked (dB above the quiet pass):")
    for f, rise, _ in best:
        print(f"  {f/1e6:10.4f} MHz   +{rise:5.1f} dB")
    if not best or best[0][1] < 12:
        print("\nNothing clearly rose. Check the walkie-talkie is ON and transmitting (its red TX light),")
        print("then try again, or add --vhf for 136-174 MHz radios.")
        return
    f = best[0][0]
    near = min(PMR446, key=lambda x: abs(x - f))
    if abs(near - f) < 8e3:
        f = near
        print(f"\nFOUND: {f/1e6:.5f} MHz = PMR446 channel {PMR446.index(near) + 1}")
    else:
        f = round(f / 6.25e3) * 6.25e3
        print(f"\nFOUND: about {f/1e6:.5f} MHz")
    # Centre 200 kHz below, so the RSPduo's own centre spike is off the signal.
    radio.ws.send(cmd("set_property", int(f - 200e3), "device_center_frequency"))
    radio.ws.send(cmd("set_property", int(f), "device_vfo_frequency"))
    radio.ws.send(cmd("set_property", "NFM", "demodulator"))
    radio.ws.send(cmd("set_property", 12500, "filter_bandwidth"))
    time.sleep(0.5)
    print("SDRconnect is now tuned to it (NFM, 12.5 kHz). Talk again: you should hear yourself.")
    (REPO / "results" / "walkie_found.json").write_text(json.dumps(
        {"frequency_hz": f, "top": [{"mhz": h[0] / 1e6, "rise_db": h[1]} for h in best]}, indent=1))


if __name__ == "__main__":
    main()
