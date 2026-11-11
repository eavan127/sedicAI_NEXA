"""
A stand-in for SDRconnect on port 5454, so the dashboard's live receiver
(RF Replay -> Receiver source "SDRconnect (live antenna)") can be developed
and demonstrated without the RSPduo. It speaks the part of WebSocket API
1.0.3 the hub uses (get_property, iq_stream_enable, set_property) and sends
the IQ stream in real time: noise, with a synthetic transmission every few
seconds (NBFM voice-like, then Morse, then AM, in turn). Synthetic only --
nothing here is a real recording.

    python scripts/fake_sdrconnect.py            # then, in the dashboard: Connect, host 127.0.0.1
    python scripts/fake_sdrconnect.py --port 5455 --fs 500000
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import analog_synth  # noqa: E402

KINDS = [("NBFM", -40e3), ("CW", 25e3), ("AM", 60e3)]


def burst(kind, offset_hz, seconds, fs, rng, snr_db=25.0):
    s, _ = analog_synth.make(kind, seconds, snr_db=200, rng=rng, offset_hz=0.0)
    w = resample_poly(s, int(fs / analog_synth.FA), 1)
    on = np.abs(w) > 1e-3
    w = w / np.sqrt(np.mean(np.abs(w[on]) ** 2) + 1e-20) * np.sqrt(10 ** (snr_db / 10) * 3000 / fs)
    return (w * np.exp(2j * np.pi * offset_hz * np.arange(len(w)) / fs)).astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5454)
    p.add_argument("--fs", type=float, default=250e3, help="IQ stream rate (must divide by 50 kS/s)")
    p.add_argument("--centre", type=float, default=145e6)
    p.add_argument("--every-s", type=float, default=8.0, help="a transmission starts this often")
    p.add_argument("--burst-s", type=float, default=3.0)
    args = p.parse_args()
    from websockets.sync.server import serve
    props = {"api_version": "1.0.3", "device_sample_rate": str(int(args.fs)), "active_device": "RSPduo (FAKE)",
             "device_center_frequency": str(int(args.centre)), "device_vfo_frequency": str(int(args.centre)),
             "demodulator": "NFM"}

    def handler(ws):
        print(f"client connected: {ws.remote_address}", flush=True)
        streaming = threading.Event()

        def control():
            try:
                for msg in ws:
                    m = json.loads(msg)
                    ev, prop = m.get("event_type"), m.get("property")
                    if ev == "get_property":
                        ws.send(json.dumps({"event_type": "get_property_response", "property": prop,
                                            "value": props.get(prop, "")}))
                    elif ev == "set_property":
                        props[prop] = m.get("value")
                        print(f"set {prop} = {m.get('value')}", flush=True)
                        ws.send(json.dumps({"event_type": "property_changed", "property": prop, "value": m.get("value")}))
                    elif ev == "iq_stream_enable":
                        (streaming.set if str(m.get("value")).lower() == "true" else streaming.clear)()
            except Exception:
                pass
        threading.Thread(target=control, daemon=True).start()
        rng = np.random.default_rng()
        step, frame_s = int(args.fs * 0.02), 0.02
        sig, sig_pos, k_burst = None, 0, 0
        next_burst = time.monotonic() + 2.0
        t0, k = time.monotonic(), 0
        try:
            while True:
                if not streaming.wait(0.5):
                    t0, k = time.monotonic(), 0
                    continue
                x = (rng.standard_normal(step) + 1j * rng.standard_normal(step)) / np.sqrt(2)
                now = time.monotonic()
                if sig is None and now >= next_burst:
                    kind, off = KINDS[k_burst % len(KINDS)]
                    sig, sig_pos, k_burst = burst(kind, off, args.burst_s, args.fs, rng), 0, k_burst + 1
                    print(f"on air: {kind} at {off / 1e3:+.0f} kHz for {args.burst_s:.0f} s", flush=True)
                    next_burst = now + args.every_s
                if sig is not None:
                    part = sig[sig_pos:sig_pos + step]
                    x[:len(part)] += part
                    sig_pos += step
                    if sig_pos >= len(sig):
                        sig = None
                x = x / 8.0
                raw = np.empty(2 * step, "<i2")
                raw[0::2] = np.clip(np.round(x.real * 32767), -32768, 32767)
                raw[1::2] = np.clip(np.round(x.imag * 32767), -32768, 32767)
                ws.send((2).to_bytes(2, "little") + raw.tobytes())
                k += 1
                rest = t0 + k * frame_s - time.monotonic()
                if rest > 0:
                    time.sleep(rest)
        except Exception:
            print("client gone", flush=True)

    with serve(handler, "127.0.0.1", args.port, max_size=None) as server:
        print(f"Fake SDRconnect on ws://127.0.0.1:{args.port} ({args.fs / 1e3:.0f} kS/s, "
              f"centre {args.centre / 1e6:.3f} MHz). Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
