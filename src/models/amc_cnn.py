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

from src.dsp import SAMPLES_PER_SYMBOL, rrc_taps


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


def _torch_normalized_cumulants(z, eps=1e-12):
    """Batched torch counterpart of src/cumulants.py's normalized_c40/
    normalized_c42/normalized_c63, returning (batch, 3) = [|C40|, |C42|,
    |C63|] for a (batch, time) complex tensor.

    This is a SEPARATE implementation from src/cumulants.py's numpy one --
    not a wrapper around it -- because it has to run batched, on a GPU, and
    inside a no-grad block during a forward pass, none of which numpy does.
    The two are checked against each other directly in
    tests/test_amc_cnn.py's test_torch_cumulants_match_numpy_cumulants:
    two implementations of the same formula WILL drift apart silently
    otherwise, so that test -- not a shared code path -- is what keeps them
    honest.

    Same normalisation as normalized_c42: scale to unit average power
    (`p`), then the reduced (approximate-circular-symmetry) cumulant
    formulas. `eps` guards the degenerate all-zero window: dividing by
    clamp_min(m2, eps) instead of raw m2 turns "0/0 -> NaN" into "0/eps ->
    0", i.e. an all-zero window produces all-zero (finite) features rather
    than crashing a batch that happens to contain one silent window.
    """
    m2 = (z.abs() ** 2).mean(dim=-1)                       # (batch,) real
    scale = torch.sqrt(m2.clamp_min(eps))
    p = z / scale.unsqueeze(-1).to(z.dtype)

    m20 = (p ** 2).mean(dim=-1)                              # (batch,) complex
    m40 = (p ** 4).mean(dim=-1)                               # (batch,) complex
    c40 = (m40 - 3.0 * m20 ** 2).abs()

    m21 = (p.abs() ** 2).mean(dim=-1)                          # (batch,) real
    m42 = (p.abs() ** 4).mean(dim=-1)                           # (batch,) real
    c42 = (m42 - 2.0 * m21 ** 2).abs()

    m63 = (p.abs() ** 6).mean(dim=-1)                           # (batch,) real
    c63 = (m63 - 9.0 * m21 * m42 + 12.0 * m21 ** 3).abs()

    return torch.stack([c40, c42, c63], dim=1)   # (batch, 3)


class CumulantFeatures(nn.Module):
    """Expert-feature branch: a fixed RRC matched filter, then three
    normalised cumulant magnitudes of the filtered complex signal --
    |C40|, |C42|, |C63| -- concatenated onto the pooled feature vector
    right before fc1.

    EXPERIMENTAL, behind `model.cumulant_features` (configs/default.yaml,
    default false -- see that flag's comment for the full measured story).
    In short: the classifier cannot distinguish 16QAM from 64QAM at all --
    51.4% single-window accuracy picking the larger class probability
    (chance), 49.7% even averaging 64 windows (still chance, because the
    error is a systematic BIAS the model's own output cannot average away),
    47.0% on true 16QAM specifically (WORSE than chance). Yet the
    information is measurably present in the samples: AUC between the two
    classes at SNR >= +2 dB, 400 windows/class, goes 0.576 (raw window) ->
    0.609 (matched-filtered window) -> 0.633 (fully recovered symbols).
    Matched filtering alone recovers most of that separation and is a fixed
    convolution -- no timing/carrier recovery needed, unlike full symbol
    recovery -- so it is cheap enough to compute inside forward() on every
    window in a batch.

    Why matched-filter-and-cumulant rather than full symbol recovery
    (src/ui/plots.py's recover_symbols) inside the model: recover_symbols
    also estimates and de-rotates a carrier offset via an FFT peak search
    and picks a timing phase by minimising amplitude spread -- both
    data-dependent, branchy, and not naturally batchable/differentiable.
    The matched filter alone buys most of the AUC (0.609 of the 0.633
    ceiling) for a fraction of the complexity: one fixed convolution.

    The matched filter is applied as a depthwise Conv1d -- groups=2, one
    real-valued kernel shared by both the real and imaginary input
    channels -- built from src.dsp.rrc_taps (SAMPLES_PER_SYMBOL), the SAME
    tap definition src/ui/plots.py's recover_symbols uses (src/dsp.py is
    the one place those taps are computed; see its module docstring). The
    kernel is a registered buffer, not a Parameter: it never appears in
    .parameters(), so it is never touched by an optimizer and adds nothing
    to a checkpoint's trainable state.

    No gradient flows through this branch at all -- the matched filter and
    the cumulant math both run inside torch.no_grad() and the output is
    .detach()-ed before being concatenated. Nothing here needs a gradient
    (the filter is fixed and the cumulant formulas involve abs() of
    near-zero complex values, which has an ill-defined/unstable
    subgradient at exactly zero); wrapping the whole branch in no_grad
    sidesteps that instead of hoping autograd handles it gracefully, while
    leaving backprop through the rest of the model (iq_branch, stft_branch,
    fc1/fc2) completely untouched.
    """

    def __init__(self):
        super().__init__()
        taps = rrc_taps(SAMPLES_PER_SYMBOL)   # numpy, unit energy, odd length
        kernel = torch.tensor(taps, dtype=torch.float32).view(1, 1, -1)
        kernel = kernel.repeat(2, 1, 1)          # one shared kernel, per-channel (groups=2)
        kernel.requires_grad_(False)
        self.register_buffer("mf_kernel", kernel)
        self.pad = kernel.shape[-1] // 2           # odd kernel -> "same"-length output
        self.out_channels = 3

    def forward(self, x):
        # x: (batch, 2, time) real/imag raw IQ.
        with torch.no_grad():
            filt = F.conv1d(x, self.mf_kernel, padding=self.pad, groups=2)
            z = torch.complex(filt[:, 0, :], filt[:, 1, :])
            feats = _torch_normalized_cumulants(z)   # (batch, 3): |C40|,|C42|,|C63|
        return feats.detach()


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024, stft_freq_summary=None,
                 cumulant_features=None):
        super().__init__()
        if stft_freq_summary is None:
            from src.config import CFG
            stft_freq_summary = CFG.get("model", {}).get("stft_freq_summary", False)
        if cumulant_features is None:
            from src.config import CFG
            cumulant_features = CFG.get("model", {}).get("cumulant_features", False)

        iq_out_channels = 128
        self.iq_branch = IQBranch(out_channels=iq_out_channels)
        self.stft_branch = STFTBranch(out_channels=64, freq_summary=stft_freq_summary)
        # Computed from what the branches actually produce, not hardcoded --
        # stft_branch.out_channels is 64 with the flag off, 67 with it on.
        fused_channels = iq_out_channels + self.stft_branch.out_channels
        self.attn_pool = AttentionPool1d(fused_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # EXPERIMENTAL, behind `model.cumulant_features` (configs/default.yaml,
        # default false). Only constructed when the flag is on: with it off,
        # this stays None and adds no parameters/buffers, so the state_dict
        # of the five checkpoints in results/ keeps loading with strict=True
        # (see CumulantFeatures' docstring for what this branch does and why).
        self.cumulant_branch = CumulantFeatures() if cumulant_features else None

        # Computed from what's actually being fused, not hardcoded -- fc1's
        # input width is fused_channels alone with the flag off, plus the 3
        # cumulant scalars (CumulantFeatures.out_channels) with it on.
        fc1_in = fused_channels + (
            self.cumulant_branch.out_channels if self.cumulant_branch is not None else 0)

        # No dummy forward pass needed -- attention pooling always outputs
        # exactly `fused_channels` values regardless of input_len. input_len
        # is kept as a parameter only so existing call sites
        # (AMC_CNN(num_classes=..., input_len=X.shape[-1])) still work
        # unchanged; it is otherwise unused.
        self.fc1 = nn.Linear(fc1_in, 256)
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

        pooled = self.attn_pool(fused, raw_power)
        if self.cumulant_branch is not None:
            cum_feats = self.cumulant_branch(x)   # (batch, 3): |C40|,|C42|,|C63|
            pooled = torch.cat([pooled, cum_feats], dim=1)

        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)
