// Core DSP building blocks: seedable PRNG, complex FFT, STFT.
//
// Does NOT reproduce numpy's PCG64 bit-for-bit -- there is no requirement to
// (each scenario is freshly randomized in the Python UI too, never replayed
// against a fixed seed for a user). What has to match exactly is the STFT
// math in stftMag(), because that feeds the ONNX model and the model's
// weights were trained against Python's torch.stft output.

// mulberry32: small, fast, good-enough statistical quality for signal
// synthesis (not cryptography, not scientific reproducibility).
export function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniform(rng, lo, hi) { return lo + rng() * (hi - lo); }
export function randint(rng, lo, hiExclusive) {
  return lo + Math.floor(rng() * (hiExclusive - lo));
}
export function choice(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }

// Box-Muller, one value per call (paired value discarded -- simplicity over
// speed; scenario lengths here are small enough that it doesn't matter).
export function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function nextPow2(n) { return 1 << Math.ceil(Math.log2(Math.max(n, 1))); }

/** In-place iterative radix-2 Cooley-Tukey FFT. re/im are Float64Arrays of
 * length N (a power of 2). invert=true computes the inverse (unnormalized
 * except for the final 1/N scale, matching numpy's ifft convention). */
export function fft(re, im, invert = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI / len) * (invert ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curWr = 1, curWi = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = re[i + j], ui = im[i + j];
        const vr = re[i + j + len / 2] * curWr - im[i + j + len / 2] * curWi;
        const vi = re[i + j + len / 2] * curWi + im[i + j + len / 2] * curWr;
        re[i + j] = ur + vr; im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
        const nWr = curWr * wr - curWi * wi;
        curWi = curWr * wi + curWi * wr;
        curWr = nWr;
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

/** Zero-padded FFT for arbitrary-length real+imag input; returns {re, im}
 * of length nextPow2(input length). Used by the barrage jammer's bandpass
 * filter, where exact length doesn't matter (it's a filter, not a model
 * input) -- only stftMag() below needs exact-length, unpadded framing. */
export function fftPadded(reIn, imIn) {
  const n = nextPow2(reIn.length);
  const re = new Float64Array(n), im = new Float64Array(n);
  re.set(reIn); im.set(imIn);
  fft(re, im, false);
  return { re, im, n };
}

export function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  return w;
}

/** STFT magnitude, framed exactly like torch.stft(..., center=False,
 * return_complex=True).abs() on a COMPLEX (non-onesided) signal -- i.e.
 * PyTorch's STFTBranch.forward(). n_fft must be a power of 2 (16 here) so
 * the plain radix-2 fft() above applies with no padding.
 *
 * iqRe/iqIm: Float64Array, the window's I and Q samples (length = window_len).
 * Returns Float64Array of length nFreq * nFrames, row-major [freq][frame] --
 * same layout the model's ONNX input expects (see model.js).
 */
export function stftMag(iqRe, iqIm, nFft, hop, window) {
  const wl = iqRe.length;
  const nFrames = 1 + Math.floor((wl - nFft) / hop);
  const out = new Float64Array(nFft * nFrames);
  const fre = new Float64Array(nFft), fim = new Float64Array(nFft);
  for (let t = 0; t < nFrames; t++) {
    const start = t * hop;
    for (let k = 0; k < nFft; k++) {
      fre[k] = iqRe[start + k] * window[k];
      fim[k] = iqIm[start + k] * window[k];
    }
    fft(fre, fim, false);
    for (let k = 0; k < nFft; k++) {
      out[k * nFrames + t] = Math.hypot(fre[k], fim[k]);
    }
  }
  return { mag: out, nFreq: nFft, nFrames };
}
