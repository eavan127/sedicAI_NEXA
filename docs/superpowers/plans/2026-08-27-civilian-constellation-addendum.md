# Civilian Constellation — Addendum Plan (Tasks 6-8)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

Extends `2026-08-26-civilian-constellation.md`, approved 2026-08-27 after live
verification of Tasks 1-5.

**Working directory:** `C:/Users/eilee/Documents/Projects/sedicAI_NEXA/.worktrees/eileen-omni-ui` (branch `eileen-omni-ui`).

## Why this addendum exists

Tasks 1-5 shipped a panel that renders, hides and captions correctly, but at
+10 dB it draws a cloud rather than clusters, so it does not yet do the job it
exists for. Measured on the running console, 4th-power phase concentration for
QPSK (1.0 = clean clusters, ~0.1 = cloud):

| stage | concentration |
|---|---|
| library capture, untouched | 0.87 |
| + scenario noise at the stated +10 dB | 0.57 |
| + the crossfade joins the scene builder uses | 0.52 |
| the window actually rendered | 0.47 |

Ruled out as causes: `SAMPLES_PER_SYMBOL = 8` is correct (0.87 at sps 8 versus
~0.4-0.5 at every other value), and carrier estimation is not the bottleneck —
sweeping every constant de-rotation within ±0.002 cyc/sample of the estimate
tops out at 0.51 on that window.

Two real causes, both upstream of the panel. Task 6 fixes the second, Task 7
the first.

---

## Task 6: Receive matched filter

**Why:** sampling an RRC-shaped signal at the symbol instant without a receive
matched filter keeps the noise of the whole 8x band while the signal occupies
only the symbol bandwidth. Measured gain on library windows: 0.62 to 0.79 at
+10 dB, 0.20 to 0.61 at +2 dB. Roll-off is not sensitive — beta 0.20 and 0.35
score the same — so guessing RadioML's exact roll-off is not a risk.

**Files:**
- Modify: `src/ui/plots.py`
- Test: `tests/test_ui_constellation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_constellation.py`, and add `rrc_taps` to the existing
`from src.ui.plots import ...` line:

```python
def _rrc_qpsk(n_symbols=64, sps=SAMPLES_PER_SYMBOL, snr_db=3.0, seed=1):
    """RRC-shaped QPSK with AWGN -- the shape a real receiver actually sees.

    The triangular pulse in _qpsk exists to test the timing search; it is the
    wrong shape for testing a matched filter, which is matched to the RRC the
    transmitter used.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                          n_symbols + 8) / np.sqrt(2)
    train = np.zeros(len(symbols) * sps, dtype=complex)
    train[::sps] = symbols
    taps = rrc_taps(sps)
    shaped = np.convolve(train, taps)[len(taps) // 2:][:n_symbols * sps]
    shaped = shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))
    noise = (rng.normal(0, 1, len(shaped))
              + 1j * rng.normal(0, 1, len(shaped))) * np.sqrt(
                  10 ** (-snr_db / 10) / 2)
    return shaped + noise


def test_rrc_taps_have_unit_energy_and_odd_length():
    taps = rrc_taps(SAMPLES_PER_SYMBOL)
    assert len(taps) % 2 == 1          # symmetric, so the filter adds no delay
    assert float(np.sum(taps ** 2)) == pytest.approx(1.0, abs=1e-9)


def test_matched_filter_tightens_clusters_a_raw_decimation_leaves_smeared():
    """The measurement that justifies this filter existing: on the same noisy
    samples, filtering before decimating pulls the constellation together."""
    z = _rrc_qpsk(snr_db=3.0)
    filtered, _, _ = recover_symbols(z)
    unfiltered = z / np.sqrt(np.mean(np.abs(z) ** 2))
    best_unfiltered = max(
        _concentration(unfiltered[phase::SAMPLES_PER_SYMBOL])
        for phase in range(SAMPLES_PER_SYMBOL))
    assert _concentration(filtered) > 0.85
    assert _concentration(filtered) > best_unfiltered + 0.15


def test_matched_filter_does_not_change_the_symbol_count():
    points, _, _ = recover_symbols(_rrc_qpsk(n_symbols=64))
    assert len(points) == 64


def test_caption_names_the_matched_filter_step():
    s = _session({"QPSK": [0.30, 0.30, 0.30, 0.95, 0.30, 0.30]})
    s.display_smoothed = False
    fig = constellation_figure(s)
    try:
        captions = " ".join(t.get_text() for t in fig.texts)
        assert "matched filter" in captions
    finally:
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ui_constellation.py -k "rrc or matched" -v`

Expected: collection error — `cannot import name 'rrc_taps' from 'src.ui.plots'`

- [ ] **Step 3: Implement**

In `src/ui/plots.py`, add after `SAMPLES_PER_SYMBOL`:

```python
# RadioML's transmitters pulse-shape with a root-raised cosine, so a receiver
# matched to it is the standard front end -- and the one this panel was
# missing. Without it a sample taken at the symbol instant carries the noise of
# the whole 8x-oversampled band while the signal occupies only the symbol
# bandwidth, which is most of why a +10 dB capture drew a cloud instead of four
# clusters (measured: 0.62 -> 0.79 4th-power phase concentration at +10 dB,
# 0.20 -> 0.61 at +2 dB).
#
# The roll-off is a guess at RadioML's, and deliberately a safe one: 0.20 and
# 0.35 score the same on those measurements, so being wrong about it costs
# nothing visible.
RRC_ROLLOFF = 0.35
RRC_SPAN_SYMBOLS = 8


def rrc_taps(sps, beta=RRC_ROLLOFF, span=RRC_SPAN_SYMBOLS):
    """Root-raised-cosine taps, unit energy, odd length so they add no delay.

    The two singular points -- t = 0 and t = 1/(4*beta) -- are written out
    separately because the general expression divides by zero at exactly those
    samples. Both branches are the limit of the closed form.
    """
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
            taps[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            taps[i] = ((np.sin(np.pi * ti * (1 - beta))
                         + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
                        / (np.pi * ti * (1 - (4 * beta * ti) ** 2)))
    return taps / np.sqrt(np.sum(taps ** 2))
```

In `recover_symbols`, apply the filter to the unit-power signal BEFORE
estimating the carrier offset — between the `z = z / np.sqrt(power)` line and
the `offset = carrier_offset(z)` line:

```python
    # Matched filter first: the carrier estimate is a 4th-power FFT peak, and
    # it finds that peak more reliably once the out-of-band noise is gone.
    z = np.convolve(z, rrc_taps(sps), mode="same")
```

Update the recovery-chain caption in `constellation_figure` to name four steps
rather than three: `unit-power scale → matched filter → de-rotate ... →
decimate ...`. The degenerate-window caption added in `9205be6` stays exactly
as it is — a window with no power still had nothing done to it.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ui_constellation.py -v`, then `python -m pytest tests -q`

Expected: 24 passed in the file; full suite up by 4 from its current 238.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ui_constellation.py src/ui/plots.py
git commit -m "feat(ui): matched-filter the window before recovering symbols"
```

---

## Task 7: Stop noising civilian emitters twice

**Why:** `civilian_library()` draws from the +10 dB bin, so those captures
already carry noise at +10 dB, and `build_scenario` then adds its own on top. A
scene labelled "+10 dB" is really about +6.9 dB. `build_dataset`'s
`build_composite_examples` already reasons about exactly this ("civilian
victims already carry their own SNR-labelled noise"); the console path does
not. This is the dominant term in the table above (0.87 to 0.57), and it means
the console misstates the SNR of every civilian scene — which affects the
detection numbers on the page, not only this panel.

**The rule to implement.** The capture ends up with ONE uniform noise floor, and
the civilian emitter's own recorded noise counts toward it.

Let `target = reference_power / 10**(snr_db / 10)` — what the code computes
today — and `carried = reference_power / 10**(library_snr_db / 10)`, the noise
the civilian recording already contains. Then:

- the floor actually used is `max(target, carried)`: you can add noise to a
  recording but never remove it, so an SNR better than the library bin is not
  achievable and must not be claimed;
- inside a civilian emitter's span, add `floor - carried`;
- everywhere else, add `floor`.

The result is a single flat noise floor across the whole capture, an effective
SNR of `min(requested, library_snr_db)`, and no silent stretches — the empty
regions still carry the same floor as the occupied ones, which both the
NOISE_FLOOR class and the occupancy readout depend on.

Scenes with no civilian emitter must come out bit-identical to today.

**Files:**
- Modify: `src/scenarios.py` — `build_scenario` gains `library_snr_db=None`
- Modify: `src/ui/session.py` — `civilian_library` exposes the bin it drew from; `load_scenario` passes it and records the achieved SNR
- Modify: `src/ui/pages/rf_replay.py` — the header states the SNR actually achieved
- Test: `tests/test_scenarios.py`, `tests/test_ui_session.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_scenarios.py`:

```python
def test_a_scene_with_no_civilian_emitter_ignores_the_library_snr():
    """The double-noise correction must be invisible to every synthetic case."""
    from src.scenarios import CASES
    a, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=0, seed=7,
                           script=CASES["All three"])
    b, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=0, seed=7,
                           script=CASES["All three"], library_snr_db=10)
    assert np.allclose(a, b)


def test_a_civilian_emitter_is_not_noised_twice():
    """The library capture already carries noise at its labelled SNR. A full
    second helping on top makes a scene labelled +10 dB really +6.9 dB."""
    from src.scenarios import CASES, CIVILIAN
    rng = np.random.default_rng(0)
    script = CASES["Civilian only"]
    lib = {c: rng.normal(0, 1, (20, 2, 512)).astype(np.float32)
           for c, _, _ in script if c in CIVILIAN}
    twice, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                               seed=7, script=script, library=lib)
    once, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                              seed=7, script=script, library=lib,
                              library_snr_db=10)
    assert np.mean(np.abs(once) ** 2) < np.mean(np.abs(twice) ** 2)


def test_the_noise_floor_stays_uniform_across_a_civilian_capture():
    """Silence outside the emitter would break the NOISE_FLOOR class and the
    occupancy readout, so the empty stretches must carry the same floor as the
    occupied ones."""
    from src.scenarios import CASES, CIVILIAN
    rng = np.random.default_rng(0)
    script = CASES["Civilian only"]          # the emitter spans 25%-75%
    lib = {c: rng.normal(0, 1, (20, 2, 512)).astype(np.float32)
           for c, _, _ in script if c in CIVILIAN}
    iq, _ = build_scenario(fs=3_200_000, total_duration=0.01, snr_db=10,
                            seed=7, script=script, library=lib,
                            library_snr_db=10)
    quiet = np.mean(np.abs(iq[:2000]) ** 2)
    occupied = np.mean(np.abs(iq[16000:18000]) ** 2)
    assert quiet > 0
    assert quiet < occupied
```

In `tests/test_ui_session.py`:

```python
def test_a_civilian_scene_reports_the_snr_it_actually_achieved(model):
    """You can add noise to a recording but never remove it, so a civilian
    scene cannot be cleaner than the library bin it was drawn from. Claiming
    the requested figure would put a number on screen that is not true of the
    capture."""
    from src.config import CFG
    s = load_scenario(model, total_duration=0.01, hop=512, snr_db=10,
                       case="Civilian only")
    assert s.snr_known is True
    assert s.true_snr_db <= max(CFG["snr_bins_db"])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_scenarios.py -k "civilian or library" -v`

Expected: `TypeError: build_scenario() got an unexpected keyword argument 'library_snr_db'`

- [ ] **Step 3: Implement**

`src/scenarios.py`: add `library_snr_db=None` to `build_scenario`. While laying
down the script, record the sample ranges belonging to civilian emitters (the
`class_name in CIVILIAN` branch already identifies them). Replace the single
`noise_power` computation with the rule stated above, and build the noise array
so civilian spans receive `floor - carried` while the rest receives `floor`.
When `library_snr_db` is None, or the script has no civilian emitter, the
behaviour must be exactly what it is today — same values, same RNG draws.

`src/ui/session.py`: `civilian_library()` already computes
`cleanest = max(CFG["snr_bins_db"])`; expose it rather than recomputing it
elsewhere. `load_scenario` passes it to `build_scenario` when the script needs
a library, and records `true_snr_db = min(snr_db, library_snr_db)` for civilian
scenes.

`src/ui/pages/rf_replay.py`: the header's SNR clause shows the achieved figure.
When the request was capped, say so in that clause rather than silently showing
a number that differs from the dropdown.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests -q`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix: civilian scenes were noised twice and misreported their SNR"
```

---

## Task 8: Re-verify in the running app

Repeat Task 5's app checks against the rebuilt console. The +10 dB `Civilian
only` case must now show four distinct clusters in the recovered panel.

If it still does not, STOP and report the measured concentration rather than
adjusting the caption to excuse the picture.
