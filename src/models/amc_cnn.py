"""
Two-branch (raw IQ + STFT) fusion CNN for Automatic Modulation Classification,
trained from scratch.

There is no pretrained backbone for raw IQ the way ImageNet exists for
images -- so "fine-tuning" here means training from random initialization.

EXPERIMENTAL BRANCH (eavan-dual-branch-fusion): this is a substantial rewrite
of the single-branch attention-pooling model on main. Breaking change -- no
existing checkpoint loads into this. Needs a full retrain and validation
(measure_variance.py) against main's numbers before being trusted for
anything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    """Energy-gated attention pooling: learned weighted pooling over time,
    where the scorer sees both the learned conv features AND the raw signal's
    instantaneous power at each timestep -- instead of a plain average/max.
    See git history on main for the full rationale (energy detection as an
    explicit cue alongside learned features).
    """

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels + 1, 1, kernel_size=1)

    def forward(self, x, raw_power):
        # x: (batch, channels, time)   raw_power: (batch, 1, time_raw)
        energy = F.adaptive_avg_pool1d(raw_power, x.shape[-1])
        combined = torch.cat([x, energy], dim=1)
        weights = torch.softmax(self.score(combined), dim=2)
        return (x * weights).sum(dim=2)


class IQBranch(nn.Module):
    """Raw-IQ path: a plain conv followed by two DILATED convs, no pooling.

    Why dilation instead of just stacking more regular layers: it widens the
    receptive field exponentially with depth instead of linearly, at no
    extra parameter cost per layer -- see the model docstring for measured
    numbers, but in short: two regular conv layers here gave each output
    position a receptive field of ~22 raw samples, smaller than a single FHSS
    hop at the current window length (4-24 hops packed into 512 samples, so
    each hop is often shorter than that). This branch's receptive field is
    ~43 samples -- roughly double, and comparable to or larger than one hop
    -- so a single position can start to span hop-to-hop structure that the
    old architecture's local receptive field could never see at all, before
    attention even gets a chance to correlate across positions.

    No pooling: dilation is already doing the "see wider" job; pooling here
    would only throw away temporal resolution the attention stage still
    needs to localise pulses/hops precisely.
    """

    def __init__(self, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.dilated1 = nn.Conv1d(64, 64, kernel_size=7, padding=6, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dilated2 = nn.Conv1d(64, out_channels, kernel_size=7, padding=12, dilation=4)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.dilated1(x)))
        x = self.relu(self.bn3(self.dilated2(x)))
        return x   # (batch, out_channels, time) -- time == input length, no pooling


class STFTBranch(nn.Module):
    """Time-frequency path: STFT computed inline from the raw IQ input, then
    a small 2D CNN over the magnitude spectrogram.

    Why a second branch instead of feeding the model spectrograms only:
    magnitude STFT discards phase, which is exactly what BPSK/QPSK/QAM need
    to be told apart (their whole identity is a phase constellation) -- see
    preprocess.py's to_spectrogram() docstring, which already documented
    this tradeoff. Conversely, phase is not what makes a chirp or a hop
    visually obvious -- frequency movement over time is, and that is exactly
    what a spectrogram makes explicit and raw IQ leaves implicit. Two
    branches means neither representation has to be the one thing that does
    both jobs.

    n_fft must be shorter than the FASTEST thing we need to resolve, not just
    "small relative to the window". FHSS hops as fast as every ~21 raw
    samples (25-150 kHz hop rate at 3.2 MHz). The original n_fft=64 was
    larger than that -- each FFT frame spanned 2-3 different hops and
    averaged their different frequencies into one blurred estimate, which
    smears away exactly the discrete jump structure that makes FHSS FHSS.
    n_fft=16 keeps every frame shorter than the fastest hop, so a frame can
    capture at most about one hop's frequency, not a blur of several.
    hop_length shrinks with it (was 16, now 4) to keep reasonable overlap
    between frames rather than leaving gaps.
    """

    def __init__(self, n_fft=16, hop_length=4, out_channels=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 2, time) real/imag -> complex signal for a proper
        # (non-onesided) STFT over the full -fs/2..+fs/2 range.
        complex_sig = torch.complex(x[:, 0, :], x[:, 1, :])
        spec = torch.stft(complex_sig, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=self.window, center=False, return_complex=True)
        mag = spec.abs().unsqueeze(1)   # (batch, 1, freq, time_frames)

        f = self.pool(self.relu(self.bn1(self.conv1(mag))))
        f = self.relu(self.bn2(self.conv2(f)))          # (batch, out_channels, freq', time')
        return f.mean(dim=2)                              # collapse frequency -> (batch, out_channels, time')


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024):
        super().__init__()
        self.iq_branch = IQBranch(out_channels=128)
        self.stft_branch = STFTBranch(out_channels=64)
        fused_channels = 128 + 64
        self.attn_pool = AttentionPool1d(fused_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # No dummy forward pass needed -- attention pooling always outputs
        # exactly `fused_channels` values regardless of input_len. input_len
        # is kept as a parameter only so existing call sites
        # (AMC_CNN(num_classes=..., input_len=X.shape[-1])) still work
        # unchanged; it is otherwise unused.
        self.fc1 = nn.Linear(fused_channels, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        raw_power = x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2   # (batch, 1, window_len)

        iq_feats = self.iq_branch(x)          # (batch, 128, time_full) -- full raw resolution
        tf_feats = self.stft_branch(x)        # (batch, 64, time_stft)  -- coarser, framed resolution

        # Fusion needs both branches on the same time axis. Previously this
        # downsampled the IQ branch DOWN to STFT's coarser resolution -- which
        # threw away exactly the fine timing IQBranch's dilated convs exist to
        # preserve, right before attention ever saw it, cancelling that
        # branch's whole purpose. Now the STFT branch is upsampled UP to the
        # IQ branch's full resolution instead, so the fine detail survives
        # into the fused sequence and into attention.
        tf_feats = F.interpolate(tf_feats, size=iq_feats.shape[-1],
                                  mode="linear", align_corners=False)
        fused = torch.cat([iq_feats, tf_feats], dim=1)   # (batch, 128+64, time_full)

        x = self.attn_pool(fused, raw_power)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
