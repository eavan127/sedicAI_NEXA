"""Export-only wrapper around AMC_CNN for ONNX / onnxruntime-web.

torch.stft on a complex tensor (STFTBranch.forward, amc_cnn.py) does not
export reliably through torch.onnx.export -- complex dtypes are a known weak
spot for ONNX exporters in general, and onnxruntime-web's browser runtime
supports a much smaller op set than the desktop one. Rather than change the
trained model, this wrapper takes the STFT magnitude as a plain real-valued
INPUT instead of computing it inside the traced graph. n_fft=16 is tiny, so
the STFT can be computed on the caller's side (numpy here for verification,
a hand-written 16-point DFT in JS for the browser) with no exporter support
needed.

Same submodules, same weights, same math -- this only moves *where* the STFT
is computed, not what it computes. compute_stft_mag() below is the reference
implementation both the export verification and (ported 1:1) the browser
side must match.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.amc_cnn import AMC_CNN, STFTBranch


class STFTBranchONNX(nn.Module):
    """STFTBranch with the internal torch.stft call replaced by an input.

    Reuses the source branch's own layers (same nn.Module objects, not
    copies) so exporting this wrapper uses the exact trained weights.
    """

    def __init__(self, stft_branch: STFTBranch):
        super().__init__()
        self.pool = stft_branch.pool
        self.conv1 = stft_branch.conv1
        self.bn1 = stft_branch.bn1
        self.conv2 = stft_branch.conv2
        self.bn2 = stft_branch.bn2
        self.relu = stft_branch.relu
        self.freq_summary = stft_branch.freq_summary
        self.summary_pool = stft_branch.summary_pool
        self.out_channels = stft_branch.out_channels

    def forward(self, mag):
        # mag: (batch, 1, freq, time_frames) -- precomputed |STFT|, same
        # shape torch.stft(...).abs().unsqueeze(1) would have produced.
        f = self.pool(self.relu(self.bn1(self.conv1(mag))))
        f = self.relu(self.bn2(self.conv2(f)))
        f = f.mean(dim=2)

        if not self.freq_summary:
            return f

        from src.models.amc_cnn import (_frequency_max, _peak_freq_delta,
                                         _spectral_flatness)
        mag_bft = mag.squeeze(1)
        freq_max = _frequency_max(mag_bft, dim=1)
        flatness = _spectral_flatness(mag_bft, dim=1)
        peak_delta = _peak_freq_delta(mag_bft, dim=1)
        extra = torch.stack([freq_max, flatness, peak_delta], dim=1)
        extra = self.summary_pool(extra)
        return torch.cat([f, extra], dim=1)


class AMC_CNN_ONNX(nn.Module):
    """AMC_CNN with the STFT magnitude taken as a second input.

    forward(iq, stft_mag) -- iq is (batch, 2, window_len), stft_mag is
    compute_stft_mag(iq, ...)'s output. Wraps an already-trained AMC_CNN
    in place; does not copy weights, so put the source model in eval() mode
    before exporting (or after -- this wrapper shares the same Parameters).
    """

    def __init__(self, model: AMC_CNN):
        super().__init__()
        self.iq_branch = model.iq_branch
        self.stft_branch = STFTBranchONNX(model.stft_branch)
        self.attn_pool = model.attn_pool
        self.relu = model.relu
        self.dropout = model.dropout
        self.fc1 = model.fc1
        self.fc2 = model.fc2

    def forward(self, iq, stft_mag):
        raw_power = iq[:, 0:1, :] ** 2 + iq[:, 1:2, :] ** 2

        iq_feats = self.iq_branch(iq)
        tf_feats = self.stft_branch(stft_mag)
        tf_feats = F.interpolate(tf_feats, size=iq_feats.shape[-1],
                                  mode="linear", align_corners=False)
        fused = torch.cat([iq_feats, tf_feats], dim=1)

        x = self.attn_pool(fused, raw_power)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def compute_stft_mag(iq: torch.Tensor, n_fft: int, hop_length: int,
                      window: torch.Tensor) -> torch.Tensor:
    """Reference STFT-magnitude computation, matching STFTBranch.forward
    exactly (same n_fft/hop_length/window, center=False, non-onesided
    complex STFT). The browser-side hand-written DFT must reproduce this
    bit-for-bit-close, not just "a" spectrogram.

    iq: (batch, 2, time) real/imag. Returns (batch, 1, freq, time_frames).
    """
    complex_sig = torch.complex(iq[:, 0, :], iq[:, 1, :])
    spec = torch.stft(complex_sig, n_fft=n_fft, hop_length=hop_length,
                       window=window, center=False, return_complex=True)
    return spec.abs().unsqueeze(1)


def stft_mag_frame_count(window_len: int, n_fft: int, hop_length: int) -> int:
    """Number of STFT frames torch.stft(..., center=False) produces --
    needed to size the dummy input used for ONNX export/tracing."""
    return 1 + (window_len - n_fft) // hop_length
