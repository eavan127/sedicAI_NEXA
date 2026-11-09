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
import math

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
        self.keep_rows = stft_branch.keep_rows
        self.summary_pool = stft_branch.summary_pool
        self.out_channels = stft_branch.out_channels
        self.n_fft = stft_branch.n_fft
        if self.keep_rows:
            self.row_proj = stft_branch.row_proj
            self.row_bn = stft_branch.row_bn

    def forward(self, mag):
        # mag: (batch, 1, freq, time_frames) -- precomputed |STFT|, same
        # shape torch.stft(...).abs().unsqueeze(1) would have produced.
        #
        # Mirrors STFTBranch.forward, including the fftshift it applies when
        # freq_summary or keep_rows is on. torch.roll by n_fft//2 IS fftshift for
        # an even n_fft, and exports to plain Slice/Concat, so the browser feeds
        # this graph the same raw-FFT-order magnitude it always did.
        if self.freq_summary or self.keep_rows:
            mag = torch.roll(mag, shifts=self.n_fft // 2, dims=2)

        f = self.pool(self.relu(self.bn1(self.conv1(mag))))
        f = self.relu(self.bn2(self.conv2(f)))
        if self.keep_rows:
            f = self.relu(self.row_bn(self.row_proj(f.flatten(1, 2))))
        else:
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


def _sorted_quantile(x, q):
    """torch.quantile(x, q, dim=1) (linear interpolation) as a sort plus two
    picks: torch.quantile itself does not export."""
    n = x.shape[1]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    srt = torch.sort(x, dim=1).values
    if frac == 0 or lo + 1 >= n:
        return srt[:, lo]
    return srt[:, lo] * (1 - frac) + srt[:, lo + 1] * frac


class PhysicsFeaturesONNX(nn.Module):
    """PhysicsFeatures (amc_cnn.py) in real arithmetic, so it exports.

    The trained branch uses torch.fft and complex tensors; ONNX has neither
    (and onnxruntime-web's op set is smaller still). This computes the same
    56 numbers the same way ExpertFeatures and SpectrumBranch already do: a
    complex value is a (real, imaginary) pair, a DFT is a matrix product with
    fixed cos / sin tables, a quantile or median is a sort, and the
    autocorrelation is summed lag by lag instead of through a 2n-point FFT.
    No parameters (the branch has none); the channelizer and the DFT tables
    are borrowed from an ExpertFeatures of the same window length, so the
    exported file carries one copy of each table, not three.

    Checked against PhysicsFeatures in tests/test_onnx_physics.py and by
    scripts/export_onnx.py on real windows. Not bit-identical: a DFT as a
    matmul rounds differently from an FFT (~1e-6 relative), which matters
    only where a feature is an argmax or a hard threshold and two candidates
    are within that rounding of each other.
    """

    def __init__(self, physics, chan):
        super().__init__()
        from src import physics_features as pf
        self.pf = pf
        self.chan = chan                 # ExpertFeatures: channelize() + cos / sin (unshifted DFT)
        self.n = physics.n
        self.han = physics.han
        self.han1 = physics.han1
        self.if_c = physics.if_c
        self.amp_c = physics.amp_c

    def _dft(self, re, im):
        return self.chan._dft(re, im)

    @staticmethod
    def _pa_db(a):
        return 10 * torch.log10((a.amax(1) + 1e-12) / (a.mean(1) + 1e-12))

    def _mag(self, re, im):
        Re, Im = self._dft(re, im)
        return torch.sqrt(Re ** 2 + Im ** 2)

    def forward(self, x):
        pf, n = self.pf, self.n
        han = self.han
        yr, yi, bins = self.chan.channelize(x)
        rms = torch.sqrt((yr ** 2 + yi ** 2).mean(1, keepdim=True)) + 1e-12
        yr, yi = yr / rms, yi / rms

        # [0:4] power-law lines, y^p by repeated squaring
        f = []
        pr, pi = yr, yi
        for p in (1, 2, 4, 8):
            if p > 1:
                pr, pi = pr * pr - pi * pi, 2 * pr * pi
            f.append(self._pa_db(self._mag(pr * han, pi * han)))

        # [4:7] symbol-rate line: one-sided spectrum of the envelope
        m1 = n // 2 + 1
        e = yr ** 2 + yi ** 2
        E = self._mag((e - e.mean(1, keepdim=True)) * han, torch.zeros_like(e))[:, :m1]
        keep = torch.ones(m1, dtype=E.dtype, device=E.device)
        keep[:3] = 0
        E = E * keep
        k = E.argmax(1)
        Ek = E.gather(1, k.unsqueeze(1)).squeeze(1)
        f += [self._pa_db(E), k.float() / n,
              10 * torch.log10((Ek + 1e-12) / (_sorted_quantile(E, 0.5) + 1e-12))]

        # [7:30] instantaneous frequency
        dr = yr[:, 1:] * yr[:, :-1] + yi[:, 1:] * yi[:, :-1]
        di = yi[:, 1:] * yr[:, :-1] - yr[:, 1:] * yi[:, :-1]
        fi = torch.atan2(di, dr) / math.pi
        # torch.median: the lower middle value (n - 1 is odd here, so the middle)
        med = torch.sort(fi, dim=1).values[:, (fi.shape[1] - 1) // 2]
        fi_c = fi - med.unsqueeze(1)
        sd = fi_c.std(1, unbiased=False) + 1e-12
        f += [sd, (fi_c ** 4).mean(1) / sd ** 4,
              (fi_c[:, 1:] - fi_c[:, :-1]).abs().mean(1) / sd]
        u = torch.clamp(fi_c / (3 * sd.unsqueeze(1)), -1, 1)
        hist_if = torch.exp(-0.5 * ((u.unsqueeze(2) - self.if_c) / pf.IF_WIDTH) ** 2).mean(1)
        g = fi_c * self.han1
        g = torch.cat([g, torch.zeros_like(g[:, :1])], dim=1)       # rfft(..., n=n): one zero
        FI = self._mag(g, torch.zeros_like(g))[:, :m1]
        FI = FI / (FI.sum(1, keepdim=True) + 1e-12)
        f_bands = [FI[:, 1:8].sum(1), FI[:, 8:32].sum(1), FI[:, 32:96].sum(1), FI[:, 96:].sum(1)]

        # [30:45] envelope
        a = torch.sqrt(yr ** 2 + yi ** 2)
        mu = a.mean(1, keepdim=True)
        var = ((a - mu) ** 2).mean(1)
        amp = [torch.sqrt(var), ((a - mu) ** 4).mean(1) / (var ** 2 + 1e-12), (a < 0.2).float().mean(1)]
        hist_a = torch.exp(-0.5 * ((a.unsqueeze(2) - self.amp_c) / pf.AMP_WIDTH) ** 2).mean(1)

        # [45:48] delay echo: |sum_t yc[t+lag] conj(yc[t])| / r0, lags 16..320
        cr, ci = yr - yr.mean(1, keepdim=True), yi - yi.mean(1, keepdim=True)
        r0 = (cr ** 2 + ci ** 2).sum(1)
        seg = []
        for lag in range(pf.LAG_MIN, pf.LAG_MAX + 1):
            re = (cr[:, lag:] * cr[:, :-lag] + ci[:, lag:] * ci[:, :-lag]).sum(1)
            im = (ci[:, lag:] * cr[:, :-lag] - cr[:, lag:] * ci[:, :-lag]).sum(1)
            seg.append(torch.sqrt(re ** 2 + im ** 2))
        seg = torch.stack(seg, dim=1) / (r0.unsqueeze(1) + 1e-12)
        j = seg.argmax(1)
        delay = [seg.amax(1), (pf.LAG_MIN + j).float() / pf.LAG_MAX,
                 torch.topk(seg, 3, dim=1).values.mean(1)]

        # [48:52] spectral shape of the raw window
        zr, zi = x[:, 0, :].float(), x[:, 1, :].float()
        Zr, Zi = self._dft(zr * han, zi * han)
        P = torch.roll(Zr ** 2 + Zi ** 2, shifts=n // 2, dims=1)    # fftshift (even n)
        Ps = F.avg_pool1d(torch.cat([P[:, -2:], P, P[:, :2]], dim=1).unsqueeze(1), 5, stride=1).squeeze(1)
        floor = _sorted_quantile(Ps, 0.3).unsqueeze(1)
        band = (Ps > 4 * floor).float()
        cnt = band.sum(1)
        logm = (torch.log(Ps + 1e-20) * band).sum(1) / torch.clamp(cnt, min=1)
        mean_in = (Ps * band).sum(1) / torch.clamp(cnt, min=1)
        flat = torch.where(cnt > 0, torch.exp(logm) / (mean_in + 1e-20), torch.ones_like(cnt))
        mid = Ps[:, 1:-1]
        peaks = ((mid > Ps[:, :-2]) & (mid > Ps[:, 2:]) & (mid > 30 * floor)).float().sum(1).clamp(max=20)
        Yr, Yi = self._dft(yr, yi)
        Y = torch.roll(Yr ** 2 + Yi ** 2, shifts=n // 2, dims=1)
        m = n // 2
        up = Y[:, m + 1:m + 64]
        dn = torch.flip(Y[:, m - 63:m], dims=[1])
        upc, dnc = up - up.mean(1, keepdim=True), dn - dn.mean(1, keepdim=True)
        den = torch.sqrt((upc ** 2).sum(1) * (dnc ** 2).sum(1))
        sym = torch.where(den > 0, (upc * dnc).sum(1) / (den + 1e-30), torch.zeros_like(den))
        spec = [bins / n, flat, peaks, sym]

        # [52:56] on/off timing of the raw window
        bp = (zr ** 2 + zi ** 2).reshape(zr.shape[0], pf.N_BLOCKS, -1).mean(2)
        on = (bp > 0.5 * bp.amax(1, keepdim=True)).float()
        onset = on.argmax(1)
        n_on = on.sum(1)
        single = ((onset > 0) & (n_on == pf.N_BLOCKS - onset.float())).float()
        flips = (on[:, 1:] - on[:, :-1]).abs().sum(1)
        timing = [10 * torch.log10(bp.amax(1) / (bp.amin(1) + 1e-12)), on.mean(1), single, flips]

        out = torch.cat([torch.stack(f, 1), hist_if, torch.stack(f_bands, 1), torch.stack(amp, 1), hist_a,
                         torch.stack(delay, 1), torch.stack(spec, 1), torch.stack(timing, 1)], dim=1)
        bad = torch.isnan(out) | torch.isinf(out)
        return torch.where(bad, torch.zeros_like(out), out)


class AMC_CNN_ONNX(nn.Module):
    """AMC_CNN with the STFT magnitude taken as a second input.

    forward(iq, stft_mag) -> (logits, attention) -- iq is
    (batch, 2, window_len), stft_mag is compute_stft_mag(iq, ...)'s output.
    Wraps an already-trained AMC_CNN in place; does not copy weights, so put
    the source model in eval() mode before exporting (or after -- this
    wrapper shares the same Parameters).

    Attention is returned as a SECOND OUTPUT rather than captured by a
    forward hook. src/timeline.py:classify_capture hooks attn_pool.score and
    softmaxes the raw scores itself; a hook has no equivalent in an exported
    graph, so the same computation is inlined here and exposed as an output.
    The Signal Analysis page plots it, and without it that page cannot exist
    outside Python.
    """

    def __init__(self, model: AMC_CNN):
        super().__init__()
        self.iq_branch = model.iq_branch
        self.stft_branch = STFTBranchONNX(model.stft_branch)
        # CumulantFeatures is a fixed depthwise conv plus mean/abs/power
        # arithmetic -- every op exports as-is, so the branch is shared rather
        # than mirrored. None when model.cumulant_features is off. Without
        # this the exported graph feeds fc1 192 inputs where the trained
        # weights expect 195, which fails at export time rather than silently.
        self.cumulant_branch = model.cumulant_branch
        # ExpertFeatures and SpectrumBranch were written for export from the
        # start: real arithmetic only, the DFT a matmul with fixed cos / sin
        # tables, no complex tensors. Shared as-is, like the cumulants. Each is
        # None when its flag is off. They were missing here until SED-39, so a
        # week-1 model with either flag on fed fc1 too few inputs.
        self.expert_branch = model.expert_branch
        self.expert_norm = model.expert_norm
        self.spectrum_branch = model.spectrum_branch
        # PhysicsFeatures uses torch.fft, so it is MIRRORED in real arithmetic
        # (PhysicsFeaturesONNX above), not shared. Its channelizer and DFT
        # tables come from the expert branch when there is one (same window
        # length, same tables), so the file holds them once.
        self.physics_branch = None
        self.physics_norm = None
        if model.physics_branch is not None:
            chan = model.expert_branch if model.expert_branch is not None else model.physics_branch.chan
            self.physics_branch = PhysicsFeaturesONNX(model.physics_branch, chan)
            self.physics_norm = model.physics_norm
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

        # AttentionPool1d.forward, inlined so the softmaxed weights can be
        # returned alongside the pooled output. Identical arithmetic --
        # verified against the hook-based path in scripts/export_onnx.py.
        energy = F.adaptive_avg_pool1d(raw_power, fused.shape[-1])
        combined = torch.cat([fused, energy], dim=1)
        weights = torch.softmax(self.attn_pool.score(combined), dim=2)
        pooled = (fused * weights).sum(dim=2)

        # Same order AMC_CNN.forward uses: the three whole-window cumulants go
        # after the pooled features, not before them.
        if self.cumulant_branch is not None:
            pooled = torch.cat([pooled, self.cumulant_branch(iq)], dim=1)
        if self.expert_branch is not None:
            pooled = torch.cat([pooled, self.expert_norm(self.expert_branch(iq))], dim=1)
        if self.spectrum_branch is not None:
            pooled = torch.cat([pooled, self.spectrum_branch(iq)], dim=1)
        if self.physics_branch is not None:
            pooled = torch.cat([pooled, self.physics_norm(self.physics_branch(iq))], dim=1)

        x = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(x), weights[:, 0, :]


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
