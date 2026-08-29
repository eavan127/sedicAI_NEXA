# Civilian Constellation — Addendum Plan 2 (Tasks 9-11)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

Follows `2026-08-27-civilian-constellation-addendum.md`. Approved 2026-08-27
after Tasks 6-7 shipped and were reviewed.

**Working directory:** `C:/Users/eilee/Documents/Projects/sedicAI_NEXA/.worktrees/eileen-omni-ui` (branch `eileen-omni-ui`).

## Why

Tasks 6 and 7 fixed the recovery chain and the double-noising, and both worked:
a bare library capture now comes through the chain at 0.76-0.83 4th-power phase
concentration, and the best window of a `Civilian only` scene reaches 0.88 —
four clean clusters.

But the panel still drew a cloud, because the SELECTOR picks the window with
the highest model confidence, and that window scored 0.31. Across the 123
windows inside that scene's civilian span: median 0.58, 27% above 0.70, 23%
below 0.40.

The cause is structural and cannot be removed. `_from_library` advances 480
samples per 512-sample library capture, so every 512-sample analysis window
straddles a seam between two unrelated recordings. A seam near the window edge
costs almost nothing (0.72-0.79 versus 0.76-0.83 for a bare capture); a seam
mid-window puts two different carrier phases in one picture and doubles the
clusters into a ring. You cannot cut a seam-free 512-sample window from
512-sample pieces.

Showing the tightest-clustering window was rejected as circular — it would pick
the window that most looks like a clean constellation and then present it as
evidence of what the modulation is. Task 9 shows four windows spread evenly
across the span instead: no selection bias, and the spread itself, including
the bad windows, is the honest picture.

---

## Task 9: Four windows, spread across the span

**Files:**
- Modify: `src/ui/session.py` (replace `best_civilian_window` with `civilian_windows`)
- Modify: `src/ui/plots.py` (`constellation_figure` becomes a 2x4 grid)
- Modify: `tests/test_ui_constellation.py`

### The selection rule

1. Pick the dominant civilian class: of `BPSK`/`QPSK`/`16QAM`/`64QAM`, the one
   whose peak probability across the capture is highest, and only if that peak
   clears the class threshold. This is the existing rule in
   `best_civilian_window` and it stays — choosing WHICH CLASS by peak
   confidence is the model's own answer, not a quality judgement.
2. Collect every window where that class clears its threshold, in time order.
3. Take `count` of them at evenly spaced positions in that list — first, last,
   and evenly between. NOT the best-looking ones, and NOT the most confident
   ones. If fewer than `count` windows qualify, return all of them.

### `CaptureSession.civilian_windows(count=4, smoothed=None)`

Returns a list of `(index, class_name, probability)` in time order, or `[]`.
Replaces `best_civilian_window` entirely — delete that method and update its
tests, rather than leaving two selectors that could disagree.

Docstring must say why the spacing is even rather than best-first: a panel that
showed the tightest-clustering window would be choosing the picture that most
looks like the answer it displays, and an operator could not tell a clean
emitter from a lucky window.

### The figure

2 rows x `count` columns, sharing nothing:
- top row: raw I/Q for each window, the exact `(2, 512)` arrays the model is fed
- bottom row: the recovered symbol points for the same windows

Per column, a short title: window index, time in ms, and that window's class
probability. Keep the class probability in `tier_color("Civilian")` — it is the
one MODEL element — and everything else in `INSTRUMENT`/`TEXT_DIM`.

Figure captions, once, at the bottom:
- the recovery chain, exactly as it reads now (four steps)
- the cluster-count caveat, as it reads now
- a new line stating the selection rule in plain words: four windows spaced
  evenly across the civilian span, not chosen for how they look, and that a
  synthesized scene splices independent recordings so some windows straddle a
  seam and will not cluster

`figsize` needs to grow — roughly `(3.2 * count, 6.4)` — and each axis keeps
`set_aspect("equal")`. Only the leftmost column carries a Y label and only the
bottom row carries X labels; repeating "I (measured)" eight times is noise.

Degenerate windows still take the existing no-power path per column: that
column's bottom title says so rather than claiming symbol points.

Returns `None` when `civilian_windows` returns `[]`, exactly as now, so the
page's visibility logic is unchanged.

### Tests

Rewrite the existing selector and figure tests against the new shape. Required
coverage:

- `civilian_windows` returns `[]` for a radar-only capture.
- It returns exactly `count` windows when many qualify, in ascending time
  order, and spans the qualifying range — first and last qualifying windows
  included.
- It returns ALL qualifying windows, without padding or repetition, when fewer
  than `count` qualify.
- It picks the strongest CLASS, not the first in `CLASSES` order (the existing
  BPSK-vs-16QAM test, adapted).
- It does NOT pick by cluster quality: construct a session where the
  qualifying windows are known, and assert the returned indices are the evenly
  spaced ones regardless of what the samples look like.
- The figure has `2 * count` axes, all square.
- Top-row axes plot 512 points each; bottom-row axes plot 64.
- Scatter points still carry `INSTRUMENT` styling, and the class text still
  carries `tier_color("Civilian")`.
- The captions name the class, the recovery chain, the 64QAM caveat, and the
  selection rule.
- A no-power window still refuses to claim recovery (adapt the existing test
  so the zeroed window is one of the four shown).

TDD as before: write the tests, run them, watch them fail, then implement.

Commit: `feat(ui): show four civilian windows spread across the span`

---

## Task 10: Review findings from Tasks 6 and 7

Five findings, from the two code-quality reviews. Do them in one commit per
area (plots, then scenarios).

### 10a — `rrc_taps` is not actually pinned by any test (Important)

`_rrc_qpsk` shapes its test signal with `rrc_taps` — the same function under
test — so the tests only prove the receive filter matches whatever `rrc_taps`
produces. The reviewer verified this: flipping a sign in the general branch
(`- 4*beta*ti*cos(...)` instead of `+`) still passes all four tests.

Add a test that pins the taps independently of the pipeline. The
ISF-free (Nyquist) criterion is the right property: convolving the RRC with
itself gives a raised cosine, which must be zero at every non-zero multiple of
`sps`. Assert that, plus symmetry (`taps == taps[::-1]`) and the peak sitting
at the centre tap. Verify the test FAILS with the sign flipped, and report the
failure output as proof — a test that passes with a corrupted filter is not a
test.

### 10b — `recover_symbols` docstring still says "Three operations" (Minor)

It performs four now. In a codebase whose whole point is not misdescribing what
the pipeline does, this one matters more than its size suggests. Fix the count
and name the matched filter in the list.

### 10c — the odd-length guarantee is accidental (Minor)

`rrc_taps` produces an odd number of taps only because `RRC_SPAN_SYMBOLS` is
even. With an odd `span` AND an odd `sps` the length is even, there is no
centre tap, the t=0 branch never fires and the "adds no delay" claim silently
becomes false. Both are exposed parameters. Raise a `ValueError` naming the
constraint, rather than documenting a trap.

### 10d — edge symbols are attenuated by `mode="same"` (Minor)

The first one or two recovered symbols are convolved against zero padding — the
reviewer measured the first symbol running about 46% low and the second about
12% low, with the tail essentially undistorted. On a 64-point scatter a couple
of points pulled toward the origin is a small but real lie about the
constellation. Drop the symbols whose filter support extends past the window
edge, and say so in the `recover_symbols` docstring. The symbol-count tests
will move; update them to the new expected count and note why in the test.

### 10e — `carried` uses the wrong emitter's power (Important, currently dormant)

`src/scenarios.py` computes `carried` from `reference_power`, which is the
FIRST NON-JAMMING emitter's power. A civilian emitter that is not first gets
SIR-scaled by up to +/-6 dB, so its actual carried noise differs from
`reference_power` — the reviewer measured a case off by ~330x. Every entry in
`CASES` happens to list the civilian class first, so this is unreachable from
the UI today, but `build_scenario` takes an arbitrary `script` and nothing
prevents a future case from ordering radar first.

Fix: track each civilian span's own post-scaling power alongside its sample
range (`emitter_powers` already holds it) and compute `carried` per span. This
also fixes the two-civilian case, where one shared top-up scalar is applied to
spans with different carried noise.

Add two tests: a script with a non-civilian emitter FIRST and a civilian second
(assert the floor is uniform across the capture), and a script with TWO
civilian emitters at different positions (assert the same).

### 10f — the header does not say what was requested (Minor)

`SNR 10.0 dB KNOWN (capped by library)` tells an operator the achieved figure
and uses a word — "library" — that appears nowhere else in the UI. Say what
was asked for and what was delivered, e.g. `SNR 10.0 dB KNOWN (capped from
20 dB)`. `CaptureSession` will need to carry the requested figure to do this.

Commits: `fix(ui): pin the RRC taps, and stop trusting the pipeline to test itself`
and `fix: compute carried noise per civilian span, not from the reference emitter`

---

## Task 11: Re-verify in the running app

Repeat the app checks. `Civilian only` at +10 dB must show four windows, at
least some of which cluster. Report the measured concentration of each of the
four rather than describing the picture.
