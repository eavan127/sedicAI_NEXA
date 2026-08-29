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


def _torch_unwrap(phase, dim=-1):
    """Batched torch counterpart of numpy.unwrap: corrects a phase sequence
    so that jumps greater than pi (wraparound at the +/-pi branch cut) are
    replaced by their 2*pi-complement equivalent, matching numpy's default
    `discont=pi` behaviour exactly.

    torch (as of the torch 2.13 this repo pins) has no `torch.unwrap` --
    unlike numpy, which has had one for years -- so this is a hand-rolled
    cumulative-correction implementation of the identical algorithm: take
    the first difference, wrap each difference into (-pi, pi], accumulate
    the correction that wrapping introduced, and add that running
    correction back onto the original phase from the second sample on.
    This is checked directly against `numpy.unwrap` on a fast-wrapping
    chirp in tests/test_amc_cnn.py's test_torch_unwrap_matches_numpy_unwrap
    -- two independent implementations of the same algorithm will drift
    apart silently otherwise, so that test, not a shared code path, is what
    keeps them honest.

    Operates along `dim` (default: the last/time axis), batched over all
    other dims, matching numpy.unwrap(..., axis=-1) applied per-row.
    """
    d = phase.diff(dim=dim)
    two_pi = 2 * torch.pi
    d_mod = torch.remainder(d + torch.pi, two_pi) - torch.pi
    # numpy's edge case: a difference of exactly -pi that wrapped from a
    # positive raw difference is pushed to +pi instead, so a real jump of
    # exactly pi is not folded back on itself.
    edge_mask = (d_mod == -torch.pi) & (d > 0)
    d_mod = torch.where(edge_mask, torch.full_like(d_mod, torch.pi), d_mod)
    correction = torch.cumsum(d_mod - d, dim=dim)

    pad_shape = list(correction.shape)
    pad_shape[dim] = 1
    zero_pad = torch.zeros(pad_shape, dtype=correction.dtype, device=correction.device)
    full_correction = torch.cat([zero_pad, correction], dim=dim)
    return phase + full_correction


def _if_spikiness_ratio(z, eps=1e-12):
    """The core `model.if_features` feature: ratio of the max absolute
    second difference of unwrapped phase to its median absolute value, for
    a (batch, time) complex tensor -> (batch,) real.

    Physical motivation (see configs/default.yaml's `if_features` comment
    and IFFeatures' docstring for the full measured story): instantaneous
    frequency is the first derivative of unwrapped phase. An LFM chirp's
    instantaneous frequency changes at a near-constant RATE (linear sweep),
    so its first derivative is smooth and its SECOND derivative is small
    and roughly uniform across the window. A frequency hopper's
    instantaneous frequency is piecewise constant with abrupt jumps, so its
    second derivative of phase is near-zero everywhere except a handful of
    spikes at the hop boundaries. The ratio of the largest such spike to
    the typical (median) value is large for a hopper and small for a
    chirp -- exactly the max-vs-median contrast that makes the ratio, not
    the raw second difference, the discriminating statistic (a few tall
    spikes barely move a median the way they dominate a max).

    `eps` guards the denominator: clamp_min(median, eps) turns a degenerate
    all-zero or perfectly-constant window (median second-difference exactly
    0) into a large-but-finite ratio instead of inf/NaN, without needing a
    branch.
    """
    phase = torch.angle(z)
    unwrapped = _torch_unwrap(phase, dim=-1)
    d1 = unwrapped.diff(dim=-1)
    d2 = d1.diff(dim=-1)
    abs_d2 = d2.abs()
    med = abs_d2.median(dim=-1).values
    mx = abs_d2.max(dim=-1).values
    return mx / med.clamp_min(eps)


class IFFeatures(nn.Module):
    """Expert-feature branch: a single scalar per window -- the
    instantaneous-frequency "spikiness" ratio (see `_if_spikiness_ratio`
    above) -- concatenated onto the pooled feature vector right before
    fc1, alongside CumulantFeatures if that flag is also on.

    EXPERIMENTAL, behind `model.if_features` (configs/default.yaml, default
    false -- see that flag's comment for the full measured story). In
    short: half of LFM_RADAR's false positives are genuinely FHSS (50.7%)
    and 35.5% of FHSS's false positives are genuinely radar -- the two
    classes share a decision boundary and trade against each other across
    fix iterations (FHSS recall rose 82.5 -> 89.7 -> 92.2 across three
    fixes while JAMMING fell 80.0 -> 73.3 -> 67.5 in the same runs). They
    differ physically in how they move through frequency: an LFM chirp
    sweeps linearly and continuously (near-constant-rate instantaneous
    frequency), while a frequency hopper holds a channel then jumps
    (piecewise-constant instantaneous frequency with abrupt spikes). The
    second derivative of unwrapped phase captures exactly that difference.

    Measured on the held-out test split, standalone windows, as the ratio
    max|d2phase| / median|d2phase|:

        SNR      radar          FHSS           AUC
        -10   3.33 +/- 0.20   3.40 +/- 0.19   0.611
         -6   3.46 +/- 0.22   3.54 +/- 0.22   0.603
         -2   3.70 +/- 0.26   4.01 +/- 0.28   0.796
         +2   4.20 +/- 0.35   5.27 +/- 0.50   0.971
         +6   4.73 +/- 0.74   6.82 +/- 1.08   0.952
        +10   5.27 +/- 1.18   9.17 +/- 1.91   0.964

    Near-perfect separation from +2 dB up, collapsing at low SNR where the
    phase derivative is noise-dominated. Pooled across all SNR the AUC is
    0.887 -- considerably stronger than CumulantFeatures' 0.609 for the
    QAM problem it addresses.

    A caution recorded here deliberately: a sibling candidate feature,
    "fraction of samples below the median", measured an AUC of exactly
    1.000 and was DISCARDED -- the fraction below a median is 0.5 by
    construction, so both classes read 0.500 +/- 0.000 and the apparent
    perfect score was a tie-breaking artifact in the ranking pipeline, not
    a real signal. Only the spikiness ratio above survived scrutiny. Two
    other candidates that WERE measured and did not ship: IF standard
    deviation, and a linear-trend correlation of instantaneous frequency
    (the latter measured a weak 0.637 AUC, and in the wrong direction) --
    neither is implemented here; shipping an unmeasured feature is exactly
    how the discarded artifact above nearly got through.

    No gradient flows through this branch: `torch.angle`'s subgradient is
    ill-defined at exactly z=0, and `.median()`/`.max()` have discontinuous
    (non-useful) gradients w.r.t. which index was selected. The whole
    branch runs inside torch.no_grad() and the output is .detach()-ed
    before concatenation, exactly like CumulantFeatures -- nothing here
    needs a gradient, and this sidesteps the ill-defined cases instead of
    hoping autograd handles them gracefully, while leaving backprop through
    the rest of the model (iq_branch, stft_branch, fc1/fc2) untouched.
    """

    def __init__(self):
        super().__init__()
        self.out_channels = 1

    def forward(self, x):
        # x: (batch, 2, time) real/imag raw IQ.
        with torch.no_grad():
            z = torch.complex(x[:, 0, :], x[:, 1, :])
            ratio = _if_spikiness_ratio(z)   # (batch,)
        return ratio.unsqueeze(1).detach()   # (batch, 1)


class AMC_CNN(nn.Module):
    def __init__(self, num_classes, input_len=1024, stft_freq_summary=None,
                 cumulant_features=None, if_features=None):
        super().__init__()
        if stft_freq_summary is None:
            from src.config import CFG
            stft_freq_summary = CFG.get("model", {}).get("stft_freq_summary", False)
        if cumulant_features is None:
            from src.config import CFG
            cumulant_features = CFG.get("model", {}).get("cumulant_features", False)
        if if_features is None:
            from src.config import CFG
            if_features = CFG.get("model", {}).get("if_features", False)

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

        # EXPERIMENTAL, behind `model.if_features` (configs/default.yaml,
        # default false). Same pattern as cumulant_branch above: only
        # constructed when the flag is on, so with it off this stays None
        # and adds no parameters/buffers -- the five checkpoints in
        # results/ keep loading with strict=True (see IFFeatures' docstring
        # for what this branch does and why). Independent of and composes
        # with cumulant_branch: either, both, or neither may be on.
        self.if_branch = IFFeatures() if if_features else None

        # Computed from what's actually being fused, not hardcoded -- fc1's
        # input width is fused_channels alone with both flags off, plus the
        # 3 cumulant scalars (CumulantFeatures.out_channels) and/or the 1 IF
        # scalar (IFFeatures.out_channels) with each respective flag on.
        fc1_in = fused_channels
        if self.cumulant_branch is not None:
            fc1_in += self.cumulant_branch.out_channels
        if self.if_branch is not None:
            fc1_in += self.if_branch.out_channels

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
        if self.if_branch is not None:
            if_feats = self.if_branch(x)   # (batch, 1): IF spikiness ratio
            pooled = torch.cat([pooled, if_feats], dim=1)

        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)
