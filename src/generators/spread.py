"""
Spread-spectrum generators for the week-1 SPREAD classes: FHSS, DSSS, LORA_CSS.

Same contract as src/generators/digital.py and the week-1 section of
src/generators/radar.py: complex64 windows of WINDOW_LEN samples at FS,
centred on 0 Hz, unit average power, AWGN at the requested SNR. No receiver
impairments here -- frequency offset, drift, fading etc. come from the shared
impairment module.

What each class means INSIDE ONE 160 us WINDOW (this project's definitions):

  class     what the window shows
  FHSS      a carrier that jumps between channels of a comb 4-24 times per
            window, carrying binary FSK data on every hop
  DSSS      continuous BPSK or QPSK data spread by a repeating m-sequence
            (length 15-127), root-raised-cosine chips
  LORA_CSS  LoRa chirps: preamble up-chirps, sync symbols, 2.25 down-chirps,
            then data up-chirps cyclically shifted by the symbol value

Scope decisions, because a 160 us window cannot show everything:

  * FHSS is FAST hopping (25-150 kHz hop rate, every channel inside the
    window's band). Real hoppers such as Bluetooth dwell 625 us on one
    channel spread over 79 MHz: one window of those shows a single GFSK burst,
    and recognising them as hoppers is the wideband detector and tracker's job.
  * LORA_CSS is limited to bandwidth / spreading-factor pairs whose symbol
    lasts at most 512 us, so a window holds at least a third of a chirp. A
    125 kHz SF7 symbol (1 ms) shows ~20 kHz of sweep in a window -- close to a
    plain tone; those settings belong to the extraction stage, which resamples
    a narrow signal so the window spans a whole chirp.
  * Kept apart from the radar classes that look alike: DSSS uses m-sequences,
    not Barker codes, and never stops (RADAR_BARKER is pulsed); FHSS never
    stops (RADAR_COSTAS hops within one pulse, then goes silent); LORA_CSS
    stays at or below 500 kHz while RADAR_FMCW sweeps 600 kHz or more.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.dsp import RRC_SPAN_SYMBOLS, rrc_taps

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]

CLASSES = ("FHSS", "DSSS", "LORA_CSS")

MAX_BANDWIDTH_HZ = 1.4e6

# --- FHSS -------------------------------------------------------------------
FHSS_HOP_RATE_HZ = (25e3, 150e3)           # 21-128 samples per hop
FHSS_SPACING_HZ = (50e3, 300e3)
FHSS_MIN_CHANNELS, FHSS_MAX_CHANNELS = 4, 32
# Data on each hop: binary continuous-phase FSK. The symbol rate is a fraction
# of the channel spacing and the tones sit within +/-0.3 spacing of the
# channel centre, so a hop never spills into its neighbours.
FHSS_SYMBOL_RATE_FRAC = (0.3, 0.6)
FHSS_MOD_INDEX = (0.5, 1.0)

# --- DSSS -------------------------------------------------------------------
# Primitive trinomials x^n + x^k + 1 -> m-sequences of length 2^n - 1.
M_SEQUENCE_TAPS = {4: 1, 5: 2, 6: 1, 7: 1}
DSSS_CHIP_SPS = (4, 6, 8, 10, 12, 14, 16)  # 800 down to 200 kchip/s
DSSS_BETA = (0.2, 0.5)

# --- LoRa -------------------------------------------------------------------
# (bandwidth Hz, spreading factor) pairs with a symbol of at most 512 us.
LORA_SETTINGS = tuple((bw, sf) for bw, sfs in ((500e3, range(5, 9)),
                                               (250e3, range(5, 8)),
                                               (125e3, range(5, 7))) for sf in sfs)
LORA_PREAMBLE = (6, 12)      # up-chirps before the sync word
LORA_DATA_SYMBOLS = (4, 16)

PARAM_KEYS = ("bandwidth_hz", "hop_rate_hz", "n_channels", "channel_spacing_hz",
              "chip_rate_hz", "code_length", "spreading_factor", "symbol_rate_hz")


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def m_sequence(degree, state=None):
    """One period (2^degree - 1 chips) of the m-sequence from the primitive
    trinomial x^degree + x^k + 1, as 0/1. `state` is the first `degree` bits
    (any nonzero start gives a cyclic shift of the same sequence)."""
    k = M_SEQUENCE_TAPS[degree]
    n = 2 ** degree - 1
    bits = list(state) if state is not None else [1] + [0] * (degree - 1)
    if len(bits) != degree or not any(bits):
        raise ValueError("state must be `degree` bits, not all zero")
    while len(bits) < n:
        i = len(bits) - degree
        bits.append(bits[i + k] ^ bits[i])
    return np.array(bits[:n], dtype=np.int8)


def _fhss(rng):
    spacing = _log_uniform(rng, *FHSS_SPACING_HZ)
    n_ch = int(rng.integers(FHSS_MIN_CHANNELS,
                            min(FHSS_MAX_CHANNELS, int(MAX_BANDWIDTH_HZ // spacing)) + 1))
    comb = (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    hop_rate = _log_uniform(rng, *FHSS_HOP_RATE_HZ)
    dwell = FS / hop_rate

    n = np.arange(WINDOW_LEN)
    hop = np.floor((n + rng.uniform(0, dwell)) / dwell).astype(int)
    n_hops = hop[-1] + 1
    channel = np.empty(n_hops, dtype=int)
    channel[0] = rng.integers(n_ch)
    for j in range(1, n_hops):                        # never stay on one channel
        channel[j] = (channel[j - 1] + rng.integers(1, n_ch)) % n_ch

    symbol_rate = spacing * rng.uniform(*FHSS_SYMBOL_RATE_FRAC)
    h = rng.uniform(*FHSS_MOD_INDEX)
    sym_idx = np.floor(n * symbol_rate / FS + rng.uniform(0, 1)).astype(int)
    data = 2 * rng.integers(0, 2, sym_idx[-1] + 1) - 1
    freq = comb[channel[hop]] + data[sym_idx] * h * symbol_rate / 2     # Hz

    # The synthesiser re-tunes at every hop, so each hop starts at its own phase.
    phase = 2 * np.pi * np.cumsum(freq / FS) + rng.uniform(0, 2 * np.pi, n_hops)[hop]
    params = dict(bandwidth_hz=n_ch * spacing, hop_rate_hz=hop_rate, n_channels=n_ch,
                  channel_spacing_hz=spacing, symbol_rate_hz=symbol_rate)
    return np.exp(1j * phase), params


def _dsss(rng):
    degree = int(rng.choice(list(M_SEQUENCE_TAPS)))
    state = rng.integers(0, 2, degree)
    if not state.any():
        state[0] = 1
    code = 1 - 2 * m_sequence(degree, state).astype(float)          # +/-1 chips
    n_code = len(code)
    sps = int(rng.choice(DSSS_CHIP_SPS))
    beta = rng.uniform(*DSSS_BETA)

    n_chips = WINDOW_LEN // sps + 2 * RRC_SPAN_SYMBOLS + 2
    k = np.arange(n_chips) + rng.integers(n_code)                     # random code phase
    if rng.random() < 0.5:
        alphabet = np.array([1, -1], dtype=complex)                   # BPSK data
    else:
        alphabet = np.exp(1j * np.pi * (np.arange(4) / 2 + 0.25))     # QPSK data
    data = alphabet[rng.integers(0, len(alphabet), k[-1] // n_code + 1)]
    chips = code[k % n_code] * data[k // n_code]

    up = np.zeros(n_chips * sps, dtype=complex)
    up[::sps] = chips
    sig = np.convolve(up, rrc_taps(sps, beta=beta), mode="same")
    edge = RRC_SPAN_SYMBOLS * sps
    sig = sig[edge:-edge]
    start = rng.integers(0, len(sig) - WINDOW_LEN + 1)
    chip_rate = FS / sps
    params = dict(bandwidth_hz=chip_rate * (1 + beta), chip_rate_hz=chip_rate,
                  code_length=n_code, symbol_rate_hz=chip_rate / n_code)
    return sig[start:start + WINDOW_LEN] * np.exp(1j * rng.uniform(0, 2 * np.pi)), params


def lora_frame(bw, sf, rng, n_preamble=8, n_data=8):
    """Instantaneous frequency (Hz, one value per sample at FS) of a LoRa frame:
    n_preamble up-chirps, 2 sync up-chirps, 2.25 down-chirps, n_data data
    up-chirps. A symbol value s cyclically shifts its up-chirp by s / 2^SF."""
    m = 2 ** sf
    ts = m / bw                                       # symbol duration, s
    segments = ([("up", 0, 1.0)] * n_preamble
                + [("up", int(rng.integers(m)), 1.0) for _ in range(2)]
                + [("down", 0, 1.0), ("down", 0, 1.0), ("down", 0, 0.25)]
                + [("up", int(rng.integers(m)), 1.0) for _ in range(n_data)])
    freq, t0 = [], 0.0
    for kind, shift, length in segments:
        t1 = t0 + length * ts
        n0, n1 = int(round(t0 * FS)), int(round(t1 * FS))
        # 0 .. length through the symbol, counted from the segment's own first
        # sample so rounding the boundary can never push it outside the band.
        u = np.arange(n1 - n0) / (ts * FS)
        if kind == "up":
            freq.append(bw * (np.mod(u + shift / m, 1.0) - 0.5))
        else:
            freq.append(-bw * (u - 0.5))
        t0 = t1
    return np.concatenate(freq)


def _lora(rng):
    bw, sf = LORA_SETTINGS[int(rng.integers(len(LORA_SETTINGS)))]
    freq = lora_frame(bw, sf, rng, n_preamble=int(rng.integers(*LORA_PREAMBLE)),
                      n_data=int(rng.integers(*LORA_DATA_SYMBOLS)))
    start = rng.integers(0, len(freq) - WINDOW_LEN + 1)
    phase = 2 * np.pi * np.cumsum(freq[start:start + WINDOW_LEN] / FS)
    params = dict(bandwidth_hz=bw, spreading_factor=sf, symbol_rate_hz=bw / 2 ** sf)
    return np.exp(1j * (phase + rng.uniform(0, 2 * np.pi))), params


_BUILDERS = {"FHSS": _fhss, "DSSS": _dsss, "LORA_CSS": _lora}


def generate_window(cls, snr_db, rng):
    """One window of spread class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests to check the physics without noise in the way.
    """
    if cls not in _BUILDERS:
        raise ValueError(f"{cls!r} is not a spread class; choose from {CLASSES}")
    sig, built = _BUILDERS[cls](rng)
    params = dict.fromkeys(PARAM_KEYS, np.nan)
    params.update(built)
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="mean")
    return sig.astype(np.complex64), params


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per parameter in PARAM_KEYS."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in PARAM_KEYS}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
