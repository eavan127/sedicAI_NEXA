"""
Radar generators.
Owner: Person A (old LFM pipeline); Jessy (week-1 radar classes).

Two parts:

  * The original LFM generator (generate_lfm_chirp_iq, embed_pulse_train,
    random_radar_example) that the deployed 8-class model, build_dataset.py,
    scenarios and the jamming sweep still use. Unchanged.
  * The week-1 radar classes (CLASSES, generate_window, generate), following
    the same contract as src/generators/digital.py -- see the section at the
    bottom of this file.

Verified by tests/test_generators.py (old LFM) and tests/test_radar_week1.py.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn


def generate_lfm_chirp_iq(fs, duration, bandwidth, f_start=None):
    """Generate a single complex-baseband LFM radar pulse.

    Phase is quadratic in time, so instantaneous frequency (its derivative)
    is linear — that linear sweep is the defining radar signature.
    """
    n = int(duration * fs)
    t = np.arange(n) / fs
    f_start = f_start if f_start is not None else -bandwidth / 2
    k = bandwidth / duration  # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.exp(1j * phase)


def embed_pulse_train(pulse, pri, fs, total_duration, time_delay=0.0, n_pulses=None):
    """Embed repeating pulses at a Pulse Repetition Interval (PRI).

    time_delay shifts where the first pulse begins. RadChar randomises this
    (1-10 us) and so must we: pulses that always start at sample 0 give the
    model a positional fingerprint rather than a signal feature.

    n_pulses caps how many pulses are emitted. RadChar sends a burst of 2-6 and
    then falls silent for the rest of the frame, whereas a continuously-scanning
    radar keeps transmitting. Both are real; None means fill the whole duration.
    """
    total_samples = int(total_duration * fs)
    pri_samples = max(int(pri * fs), 1)
    offset = int(time_delay * fs)

    out = np.zeros(total_samples, dtype=complex)
    emitted = 0
    for start in range(offset, max(total_samples - len(pulse), 1), pri_samples):
        if n_pulses is not None and emitted >= n_pulses:
            break
        out[start:start + len(pulse)] += pulse
        emitted += 1
    return out


def random_radar_example(fs=None, total_duration=None, rng=None):
    """One randomized radar example, parameters drawn from the config ranges.

    Randomizing per example prevents the model from memorizing one specific
    radar signature instead of learning the general LFM concept.
    """
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]

    cfg = CFG["radar"]
    pulse_width = rng.uniform(*cfg["pulse_width_s"])
    bandwidth = rng.uniform(*cfg["bandwidth_hz"])
    time_delay = rng.uniform(*cfg["time_delay_s"])

    # PRI is drawn CONDITIONAL on pulse width, so pulse_width < PRI always.
    # Sampling them independently allows duty > 100%, i.e. a pulse starting
    # before the previous one ended — which is physically impossible and
    # produces overlapping garbage.
    pri_lo = max(cfg["pri_s"][0], pulse_width / cfg["max_duty_cycle"])
    pri_hi = max(pri_lo, cfg["pri_s"][1])
    # LOG-uniform, not uniform. PRI spans two orders of magnitude (67 us to
    # 10 ms), and drawing uniformly puts ~95% of examples above 1 ms — so the
    # window almost never contains more than one pulse and the pulse-train
    # structure disappears from the training data. Log-uniform gives each
    # decade equal representation.
    pri = float(np.exp(rng.uniform(np.log(pri_lo), np.log(pri_hi))))
    # Randomize sweep direction (up-chirp vs down-chirp)
    f_start = -bandwidth / 2 if rng.random() > 0.5 else bandwidth / 2
    bandwidth = bandwidth if f_start < 0 else -bandwidth

    # Two emission patterns, both real. RadChar sends a short burst (2-6 pulses)
    # then falls silent; a continuously-scanning radar keeps transmitting. We
    # cover both, since the organisers' stream could resemble either.
    n_pulses = (int(rng.integers(*cfg["n_pulses"]))
                if rng.random() < cfg["burst_fraction"] else None)

    pulse = generate_lfm_chirp_iq(fs, pulse_width, bandwidth, f_start)
    return embed_pulse_train(pulse, pri, fs, total_duration, time_delay, n_pulses)


# ---------------------------------------------------------------------------
# Week-1 radar classes (README "Shared data workflow", dataset_classes)
#
# One window = WINDOW_LEN samples at FS, built directly at that length, centred
# on 0 Hz. Placement in the band, frequency offset, drift and every other
# receiver effect come from the shared impairment module, not from here.
#
#   class              what is inside one pulse
#   RADAR_PULSE        nothing: the carrier switched on and off
#   LFM_RADAR          frequency sweeps linearly across the bandwidth
#   RADAR_BARKER       binary (0 / pi) phase code, Barker lengths 2-13
#   RADAR_POLY_BARKER  polyphase code with every sidelobe <= 1 (POLY_BARKER_PHASES)
#   RADAR_FRANK        M x M polyphase code: a staircase of M frequency steps
#   RADAR_COSTAS       N frequency hops, each frequency once, in Costas order
#   RADAR_FMCW         continuous sawtooth or triangle sweep, no silence
#   RADAR_CW           continuous unmodulated carrier, no silence
#
# Pulsed classes: every window holds at least one WHOLE pulse (a window
# labelled radar that shows only noise teaches nothing), at a random position,
# with neighbours at the pulse repetition interval cut off by the window edge.
# Duty cycle stays at or below radar.max_duty_cycle (15%): above it a chirp
# train is genuinely the same shape as frequency hopping (configs/default.yaml
# has the measurement). RadChar covers the 44-94% regime with its own windows.
#
# Power and SNR: pulses are scaled to unit power WHILE ON, and noise is added
# against that ("active" reference, the same as preprocess.add_awgn's
# default), so the SNR label means "SNR during the pulse". Scaling to unit
# average power over the window instead would make a 1% duty radar 20 dB
# louder than its label claims during the pulse.
# ---------------------------------------------------------------------------

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]

CLASSES = ("RADAR_PULSE", "LFM_RADAR", "RADAR_BARKER", "RADAR_POLY_BARKER",
           "RADAR_FRANK", "RADAR_COSTAS", "RADAR_FMCW", "RADAR_CW")
CONTINUOUS_CLASSES = ("RADAR_FMCW", "RADAR_CW")
PULSED_CLASSES = tuple(c for c in CLASSES if c not in CONTINUOUS_CLASSES)

# Highest occupied bandwidth any class may use. Nyquist is 1.6 MHz; the
# impairment module still has to place the signal somewhere in the band.
MAX_BANDWIDTH_HZ = 1.4e6
# Longest pulse, so one whole pulse always fits with room to move it.
MAX_PULSE_SAMPLES = 448
# Shortest silence between pulses: there must be a visible gap.
MIN_GAP_SAMPLES = 8
DUTY_RANGE = (0.01, CFG["radar"]["max_duty_cycle"])
# Code chip length: 3 samples (0.9 us, ~1.07 MHz wide) to 24 (7.5 us).
CHIP_SAMPLES = (3, 24)
PULSE_WIDTH_S = (1e-6, 100e-6)
LFM_PULSE_WIDTH_S = (5e-6, 100e-6)
LFM_BANDWIDTH_HZ = (100e3, MAX_BANDWIDTH_HZ)
FMCW_PERIOD_SAMPLES = (64, WINDOW_LEN)
# Starts at 600 kHz so it never overlaps LORA_CSS (src/generators/spread.py,
# 500 kHz at most): a LoRa preamble is a train of identical up-chirps, the
# same shape as a sawtooth FMCW sweep, so the two classes are kept apart by
# bandwidth.
FMCW_BANDWIDTH_HZ = (600e3, MAX_BANDWIDTH_HZ)
# A CW radar is one tone. Generated a few kHz off 0 Hz so it is never pure DC
# (which window normalisation would flatten); the impairment module places it.
CW_OFFSET_HZ = (1e3, 20e3)
# Rise/fall time of each pulse edge, in samples: real transmitters do not
# switch on in zero time.
EDGE_SAMPLES = (0, 3)

BARKER_CODES = {
    2: (1, -1),
    3: (1, 1, -1),
    4: (1, 1, -1, 1),
    5: (1, 1, 1, -1, 1),
    7: (1, 1, 1, -1, -1, 1, -1),
    11: (1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1),
    13: (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1),
}

# Polyphase Barker codes: unit-modulus sequences whose aperiodic
# autocorrelation sidelobes all have magnitude <= 1 (the Barker property,
# with any phase allowed instead of only 0 / pi). Found by numerical search
# (random restarts, minimising the sum of sidelobe magnitudes ** 12), purely
# binary solutions rejected; tests/test_radar_week1.py re-checks the property
# for every entry. Phases in radians, first chip 0.
POLY_BARKER_PHASES = {
    5: (0.000000, 1.663773, 0.867752, 0.876215, 4.708070),
    6: (0.000000, 0.622428, 0.197658, 5.008876, 2.489711, 5.206534),
    7: (0.000000, 5.678777, 3.217955, 2.849339, 4.622043, 2.203752, 4.212176),
    8: (0.000000, 3.866672, 2.671285, 5.722893, 1.641059, 2.992153, 2.307051, 2.843079),
    9: (0.000000, 3.197410, 0.957704, 3.797340, 4.592651, 0.479943, 0.439571, 5.614362, 5.514023),
    10: (0.000000, 1.159890, 5.804668, 0.798107, 3.749316, 4.365322, 4.437885, 1.630155,
         6.124581, 4.117913),
    11: (0.000000, 4.506610, 2.577681, 0.724114, 1.814363, 0.571693, 1.489250, 0.918417,
         2.660156, 4.616728, 0.102784),
    12: (0.000000, 4.758899, 4.026112, 4.160728, 3.502891, 1.134055, 3.987885, 3.929915,
         5.498640, 0.661137, 4.613130, 0.912532),
    13: (0.000000, 4.861908, 3.516266, 0.970845, 4.626875, 0.427998, 4.650088, 5.737954,
         4.270538, 5.137199, 0.248855, 1.730007, 2.826237),
    14: (0.000000, 3.150310, 0.141411, 0.962345, 3.781566, 0.971424, 0.069950, 1.773022,
         2.591641, 2.622825, 0.061734, 5.761290, 5.575541, 4.764183),
    15: (0.000000, 2.823142, 6.105272, 4.823022, 1.722396, 3.867602, 5.159236, 2.273161,
         1.617734, 0.671265, 5.103652, 5.579130, 0.254281, 0.524431, 0.990560),
    16: (0.000000, 0.909245, 5.157299, 5.798300, 0.029784, 0.817913, 5.012100, 6.215184,
         2.800975, 2.179970, 1.978983, 5.083637, 4.521295, 0.968874, 5.338298, 2.357037),
    17: (0.000000, 3.939472, 1.231008, 0.256895, 0.473355, 4.714688, 1.395086, 1.056457,
         1.878211, 0.385253, 0.701778, 3.659886, 5.276332, 4.447579, 5.546676, 1.228814,
         3.026862),
    18: (0.000000, 4.153422, 0.608277, 4.222968, 1.589680, 3.438831, 2.118289, 1.566543,
         5.839374, 4.485541, 4.346135, 6.204485, 5.414540, 5.845864, 1.061440, 3.836982,
         4.955358, 0.201621),
    19: (0.000000, 0.679922, 1.152128, 6.049969, 0.077806, 6.053983, 1.228461, 3.146339,
         4.303742, 1.543809, 3.915288, 0.764812, 5.671658, 1.548327, 4.962321, 3.307754,
         2.301462, 0.212725, 4.994940),
    20: (0.000000, 5.527499, 0.292891, 0.391563, 1.529710, 2.509554, 1.533921, 5.727662,
         0.379407, 5.843846, 3.951069, 3.517859, 0.590137, 4.858499, 2.570191, 4.069753,
         0.771633, 2.099692, 4.302722, 1.099137),
    21: (0.000000, 3.822594, 2.358602, 0.886237, 4.607787, 3.747014, 1.329945, 2.523858,
         0.378766, 1.142904, 4.711899, 5.941272, 5.437661, 1.907254, 3.885329, 4.233690,
         5.528597, 4.390202, 5.901587, 4.898460, 0.236121),
    22: (0.000000, 4.916971, 2.159072, 0.832306, 3.333414, 2.403002, 5.713381, 4.934781,
         5.296562, 3.146175, 1.467551, 2.353720, 1.242318, 2.980731, 1.223645, 1.901236,
         0.939610, 2.856511, 4.573292, 0.632187, 2.526817, 4.424609),
    23: (0.000000, 5.131874, 2.112480, 0.594717, 3.952138, 3.062465, 2.942717, 2.232188,
         3.403283, 2.923234, 4.190767, 1.328099, 6.071974, 1.488452, 4.125882, 3.429977,
         3.573312, 0.532741, 2.230787, 5.205836, 0.023246, 2.506141, 3.764224),
    24: (0.000000, 4.792308, 3.348242, 2.690994, 1.187178, 5.145249, 3.057063, 5.884134,
         3.735374, 2.335271, 3.449596, 3.137374, 3.097493, 3.259438, 4.038588, 6.053555,
         0.729722, 1.572291, 4.330270, 6.193251, 2.979437, 5.628167, 2.032724, 3.877389),
}

FRANK_M = (3, 4, 5, 6, 7, 8)            # codes of 9 ... 64 chips
# Welch construction: for prime p with primitive root g, g^(i+c) mod p for
# i = 1 .. p-1 is a Costas sequence of length p-1.
COSTAS_PRIMES = {5: 2, 7: 3, 11: 2, 13: 2}

PARAM_KEYS = ("pulse_width_us", "pri_us", "duty", "bandwidth_hz",
              "code_length", "chip_samples", "n_full_pulses")


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _chip(rng, n_chips):
    hi = min(CHIP_SAMPLES[1], MAX_PULSE_SAMPLES // n_chips)
    return int(rng.integers(CHIP_SAMPLES[0], hi + 1))


def _phase_code(phases, chip):
    return np.repeat(np.exp(1j * np.asarray(phases, dtype=float)), chip)


def frank_phases(m):
    """Frank code of M*M chips: phase 2*pi*i*j/M for group i, chip j. Within
    group i the phase advances by 2*pi*i/M per chip -- a constant frequency
    that steps up from group to group."""
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    return (2 * np.pi * i * j / m).ravel()


def costas_sequence(p, shift=0):
    """Welch Costas sequence of length p-1, values 1 .. p-1, each once."""
    g = COSTAS_PRIMES[p]
    return np.array([pow(g, i + shift, p) for i in range(1, p)])


def _edges(pulse, rng):
    edge = int(rng.integers(EDGE_SAMPLES[0], EDGE_SAMPLES[1] + 1))
    if edge and len(pulse) > 2 * edge:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(1, edge + 1) / (edge + 1)))
        pulse = pulse.copy()
        pulse[:edge] *= ramp
        pulse[-edge:] *= ramp[::-1]
    return pulse


def _pulse(cls, rng):
    """One pulse of class `cls` at 0 Hz: (complex pulse, params)."""
    p = dict.fromkeys(PARAM_KEYS, np.nan)
    if cls == "RADAR_PULSE":
        pw = max(CHIP_SAMPLES[0], int(round(_log_uniform(rng, *PULSE_WIDTH_S) * FS)))
        pulse = np.ones(pw, dtype=complex)
        p["bandwidth_hz"] = FS / pw
    elif cls == "LFM_RADAR":
        pw = int(round(rng.uniform(*LFM_PULSE_WIDTH_S) * FS))
        bw = rng.uniform(*LFM_BANDWIDTH_HZ) * rng.choice((-1, 1))   # up or down
        t = np.arange(pw) / FS
        pulse = np.exp(2j * np.pi * (-bw / 2 * t + 0.5 * bw / (pw / FS) * t ** 2))
        p["bandwidth_hz"] = abs(bw)
    elif cls == "RADAR_BARKER":
        code = np.array(BARKER_CODES[int(rng.choice(list(BARKER_CODES)))])
        chip = _chip(rng, len(code))
        pulse = _phase_code(np.where(code > 0, 0.0, np.pi), chip)
        p.update(code_length=len(code), chip_samples=chip, bandwidth_hz=FS / chip)
    elif cls == "RADAR_POLY_BARKER":
        phases = np.array(POLY_BARKER_PHASES[int(rng.choice(list(POLY_BARKER_PHASES)))])
        if rng.random() < 0.5:          # the conjugate and the time-reverse keep
            phases = -phases            # the sidelobe property; vary the codes
        if rng.random() < 0.5:
            phases = phases[::-1]
        chip = _chip(rng, len(phases))
        pulse = _phase_code(phases, chip)
        p.update(code_length=len(phases), chip_samples=chip, bandwidth_hz=FS / chip)
    elif cls == "RADAR_FRANK":
        m = int(rng.choice(FRANK_M))
        chip = _chip(rng, m * m)
        pulse = _phase_code(frank_phases(m), chip)
        p.update(code_length=m * m, chip_samples=chip, bandwidth_hz=FS / chip)
    elif cls == "RADAR_COSTAS":
        prime = int(rng.choice(list(COSTAS_PRIMES)))
        seq = costas_sequence(prime, shift=int(rng.integers(0, prime - 1)))
        n = len(seq)
        # Sub-pulse of T samples, hop spacing 1/T: total width n/T * FS.
        hop = int(rng.integers(int(np.ceil(n * FS / MAX_BANDWIDTH_HZ)), MAX_PULSE_SAMPLES // n + 1))
        freq = np.repeat((seq - (n + 1) / 2) / hop, hop)           # cycles per sample
        pulse = np.exp(2j * np.pi * np.cumsum(freq))
        p.update(code_length=n, chip_samples=hop, bandwidth_hz=n * FS / hop)
    else:
        raise ValueError(f"{cls!r} is not a pulsed radar class")
    p["pulse_width_us"] = len(pulse) / FS * 1e6
    return _edges(pulse, rng), p


def _pulse_train(pulse, rng):
    """Fill a window with copies of `pulse` at a random pulse repetition
    interval, one whole copy at a random position. Returns (window, pri, n_full)."""
    pw = len(pulse)
    duty = _log_uniform(rng, *DUTY_RANGE)
    pri = max(pw + MIN_GAP_SAMPLES, int(round(pw / duty)))
    start = int(rng.integers(0, WINDOW_LEN - pw + 1))
    while start - pri + pw > 0:      # walk back to the first pulse that shows
        start -= pri
    out = np.zeros(WINDOW_LEN, dtype=complex)
    n_full = 0
    for s in range(start, WINDOW_LEN, pri):
        lo, hi = max(s, 0), min(s + pw, WINDOW_LEN)
        out[lo:hi] = pulse[lo - s:hi - s]
        n_full += (s >= 0 and s + pw <= WINDOW_LEN)
    return out, pri, n_full


def _continuous(cls, rng):
    p = dict.fromkeys(PARAM_KEYS, np.nan)
    n = np.arange(WINDOW_LEN)
    if cls == "RADAR_FMCW":
        period = int(rng.integers(FMCW_PERIOD_SAMPLES[0], FMCW_PERIOD_SAMPLES[1] + 1))
        bw = rng.uniform(*FMCW_BANDWIDTH_HZ)
        pos = ((n + rng.integers(0, period)) % period) / period    # 0 .. 1 in each sweep
        if rng.random() < 0.5:
            shape = pos                                   # sawtooth
        else:
            shape = 1 - np.abs(2 * pos - 1)               # triangle
        if rng.random() < 0.5:
            shape = 1 - shape                             # sweep down instead of up
        freq = (shape - 0.5) * bw / FS                     # cycles per sample
        sig = np.exp(2j * np.pi * np.cumsum(freq))
        p.update(bandwidth_hz=bw, chip_samples=period)
    elif cls == "RADAR_CW":
        f = rng.uniform(*CW_OFFSET_HZ) * rng.choice((-1, 1))
        sig = np.exp(2j * np.pi * f / FS * n)
        p["bandwidth_hz"] = abs(f)
    else:
        raise ValueError(f"{cls!r} is not a continuous radar class")
    p["duty"] = 1.0
    return sig, p


def _active_power(sig):
    mag_sq = np.abs(sig) ** 2
    return mag_sq[mag_sq > 0.01 * mag_sq.max()].mean()


def generate_window(cls, snr_db, rng):
    """One window of radar class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit power while the radar is on) --
    used by the tests to check the physics without noise in the way.
    """
    if cls in PULSED_CLASSES:
        pulse, params = _pulse(cls, rng)
        sig, pri, n_full = _pulse_train(pulse, rng)
        params.update(pri_us=pri / FS * 1e6, duty=len(pulse) / pri, n_full_pulses=n_full)
    elif cls in CONTINUOUS_CLASSES:
        sig, params = _continuous(cls, rng)
    else:
        raise ValueError(f"{cls!r} is not a radar class; choose from {CLASSES}")
    sig = sig * np.exp(1j * rng.uniform(0, 2 * np.pi)) / np.sqrt(_active_power(sig))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="active")
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
