"""Generates a fixed test IQ signal and its STFT magnitude via
compute_stft_mag() (the ONNX export's reference implementation, itself
verified to match torch.stft inside the real model). JS's stftMag() in
dsp.js must reproduce this exactly -- see stft_check.mjs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.models.onnx_export import compute_stft_mag

WINDOW_LEN = 512
N_FFT, HOP = 16, 4

t = np.arange(WINDOW_LEN)
iq_re = np.cos(0.05 * t) + 0.3 * np.sin(0.31 * t)
iq_im = np.sin(0.07 * t) - 0.2 * np.cos(0.13 * t)
iq = torch.tensor(np.stack([iq_re, iq_im])[None, :, :], dtype=torch.float32)

window = torch.hann_window(N_FFT)
mag = compute_stft_mag(iq, N_FFT, HOP, window)[0, 0].numpy()  # (freq, frames)

out = {
    "iq_re": iq_re.tolist(),
    "iq_im": iq_im.tolist(),
    "n_fft": N_FFT,
    "hop": HOP,
    "n_freq": mag.shape[0],
    "n_frames": mag.shape[1],
    "mag": mag.tolist(),
}
out_path = Path(__file__).parent / "stft_reference.json"
out_path.write_text(json.dumps(out))
print(f"wrote {out_path}")
