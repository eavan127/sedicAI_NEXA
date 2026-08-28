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


def _frequency_max(mag, dim=1):
    """Per-frame max over frequency. A narrowband victim survives a
    broadband floor in the max where it would be erased in the mean -- the
    direct counter to "victim buried under jammer"."""
    return mag.max(dim=dim).values


def _spectral_flatness(mag, dim=1, eps=1e-8):
    """Geometric mean over frequency divided by arithmetic mean over
    frequency. Near 1 for broadband noise (energy spread evenly -> geometric
    and arithmetic mean converge), near 0 for tonal/narrowband energy (a
    handful of near-zero bins crush the geometric mean toward zero while the
    arithmetic mean barely moves). Scale-invariant -- multiplying the whole
    spectrum by a loud jammer's gain does not change the ratio -- so a loud
    jammer does not swamp it. The log is guarded with eps against true
    zeros."""
    log_mag = torch.log(mag + eps)
    gmean = torch.exp(log_mag.mean(dim=dim))
    amean = mag.mean(dim=dim)
    return gmean / (amean + eps)


def _peak_freq_delta(mag, dim=1):
    """Frame-to-frame change in argmax over frequency, normalised to
    [-1, 1] by the number of frequency bins. This is hopping, stated
    directly: FHSS visiting bin 2 then bin 6 shows up as a nonzero delta,
    where the frequency-mean collapse could never distinguish that from a
    stationary tone at bin 4. The first frame has no predecessor, so its
    delta is defined as 0."""
    n_bins = mag.shape[dim]
    peak = mag.argmax(dim=dim).to(mag.dtype)   # (..., time_frames)
    delta = torch.zeros_like(peak)
    delta[..., 1:] = (peak[..., 1:] - peak[..., :-1]) / n_bins
    return delta


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

    EXPERIMENTAL, behind `model.stft_freq_summary` (configs/default.yaml,
    default false): even with n_fft fixed, the branch's *own* final line
    used to throw the frequency axis away regardless -- `f.mean(dim=2)`
    means "peak at bin 2 then bin 6" (FHSS) and "peak at bin 4 then bin 4"
    (tone jamming) end up as identical features. Measured consequences:
    FHSS recall 0.960 -> 0.040 as jammer-to-signal ratio goes 0dB -> +10dB;
    46.5% of held-out jamming predicted as FHSS; jamming's seed spread 10.8
    points against FHSS's 1.1.

    There is also a resolution problem the mean-pool compounds: n_fft=16 at
    fs=3.2 MHz gives 200 kHz per bin, and the original 2x2 MaxPool halves
    that to 400 kHz. An FHSS hop between adjacent channels is 10-48 kHz --
    a twentieth of a bin -- so pooling over frequency throws away
    resolution this branch cannot spare.

    When the flag is true: the MaxPool2d(2) becomes a (1, 2) pool -- time
    only, frequency untouched, so frequency resolution stays at 200 kHz/bin
    -- and three explicit per-frame features are computed directly from the
    STFT MAGNITUDE (not the conv feature maps -- the spectral flatness of a
    learned feature map is not spectral flatness) and concatenated as extra
    channels onto the frequency-collapsed conv features: frequency max,
    spectral flatness, and peak-frequency delta (see the module-level
    helpers above for what each one is and why). AttentionPool1d and
    everything downstream of this branch's channel count are otherwise
    untouched -- when the flag is false this class computes exactly what it
    always has, byte for byte, so the checkpoints in results/ keep loading.
    """

    def __init__(self, n_fft=16, hop_length=4, out_channels=64, freq_summary=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_summary = freq_summary
        self.register_buffer("window", torch.hann_window(n_fft))

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d((1, 2)) if freq_summary else nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Downsamples the 3 hand-built per-frame features from the raw STFT
        # frame count down to the conv path's pooled time axis, so the two
        # can be concatenated. Same factor-of-2 the conv path applies in
        # time, so both land on the same time' count.
        self.summary_pool = nn.AvgPool1d(2) if freq_summary else None

        # out_channels is fixed regardless of the flag; the extra 3 hand-built
        # channels only exist when freq_summary is on. Read by AMC_CNN to
        # size fusion correctly instead of hardcoding it.
        self.out_channels = out_channels + (3 if freq_summary else 0)

    def forward(self, x):
        # x: (batch, 2, time) real/imag -> complex signal for a proper
        # (non-onesided) STFT over the full -fs/2..+fs/2 range.
        complex_sig = torch.complex(x[:, 0, :], x[:, 1, :])
        spec = torch.stft(complex_sig, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=self.window, center=False, return_complex=True)
        mag = spec.abs().unsqueeze(1)   # (batch, 1, freq, time_frames)

        f = self.pool(self.relu(self.bn1(self.conv1(mag))))
        f = self.relu(self.bn2(self.conv2(f)))          # (batch, out_channels, freq', time')
        f = f.mean(dim=2)                                 # collapse frequency -> (batch, out_channels, time')

        if not self.freq_summary:
            return f

        mag_bft = mag.squeeze(1)   # (batch, freq, time_frames)
        freq_max = _frequency_max(mag_bft, dim=1)          # (batch, time_frames)
        flatness = _spectral_flatness(mag_bft, dim=1)       # (batch, time_frames)
        peak_delta = _peak_freq_delta(mag_bft, dim=1)        # (batch, time_frames)

        extra = torch.stack([freq_max, flatness, peak_delta], dim=1)  # (batch, 3, time_frames)
        extra = self.summary_pool(extra)                                # -> (batch, 3, time')

        return torch.cat([f, extra], dim=1)   # (batch, out_channels+3, time')


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024, stft_freq_summary=None):
        super().__init__()
        if stft_freq_summary is None:
            from src.config import CFG
            stft_freq_summary = CFG.get("model", {}).get("stft_freq_summary", False)

        iq_out_channels = 128
        self.iq_branch = IQBranch(out_channels=iq_out_channels)
        self.stft_branch = STFTBranch(out_channels=64, freq_summary=stft_freq_summary)
        # Computed from what the branches actually produce, not hardcoded --
        # stft_branch.out_channels is 64 with the flag off, 67 with it on.
        fused_channels = iq_out_channels + self.stft_branch.out_channels
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
