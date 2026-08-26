"""One loaded capture, shared across every page.

Populated once by a load action on RF Replay and stored in gr.State. Every
other page reads it. No page re-runs inference -- a 0.1 s capture at hop 256
is ~1,250 forward passes, and doing that per tab switch would make the console
unusable.

This layer is where the DISPLAY-ONLY rules get turned on. src/timeline.py
defaults them off so it stays a primitive the scorecard path can rely on; the
UI opts in here, and only here.
"""
from dataclasses import dataclass, field

import numpy as np

from src.config import CFG, CLASSES, resolve_multilabel_thresholds
from src.measure import noise_floor_power
from src.scenarios import CASES, CIVILIAN, build_scenario

_CIVILIAN_LIBRARY = None


def civilian_library():
    """Real RadioML captures per civilian class, drawn from the TRAIN split.

    Train, not test: the console is a demonstration surface, and putting
    held-out evaluation data on screen invites exactly the confusion this
    project has been careful to avoid. The test split stays reserved for
    measurement.

    Loaded once and cached -- reading 320 MB per scenario would make the
    console unusable.
    """
    global _CIVILIAN_LIBRARY
    if _CIVILIAN_LIBRARY is None:
        import numpy as _np
        from src.train import load_data, stratified_split
        X, y, snr = load_data()
        d = CFG["dataset"]
        train, _, _ = stratified_split(y, snr, d["val_frac"], d["test_frac"],
                                        d["seed"])
        # Draw from the HIGHEST SNR bin only. These windows already carry
        # noise at their labelled SNR, and build_scenario adds its own on top
        # -- so a -10 dB capture plus scenario noise is unrecoverable, and the
        # scene's stated SNR would be a fiction. build_dataset solves the same
        # problem for its composites via radioml_clean_min_snr_db: use the
        # cleanest civilian available, then noise it once.
        cleanest = max(CFG["snr_bins_db"])
        lib = {}
        for cls in CIVILIAN:
            j = CLASSES.index(cls)
            # standalone windows only -- a composite window would drag a
            # second emitter into the scene unannounced
            sel = train[(y[train][:, j] > 0.5) & (y[train].sum(axis=1) == 1)
                         & (snr[train] == cleanest)]
            if len(sel):
                lib[cls] = _np.asarray(X[sel[:400]])
        _CIVILIAN_LIBRARY = lib
    return _CIVILIAN_LIBRARY
from src.timeline import classify_capture, detections, smooth, tier_track

MAX_WINDOWS = 4000

# Defaults for the display-layer rules. See the spec's "Deployment-layer
# detection rules" section for the measurements behind these.
DEFAULT_NOISE_GATE = 0.5
DEFAULT_HOLD_US = 3000.0
DEFAULT_ALPHA = 0.3


@dataclass
class CaptureSession:
    iq: np.ndarray          # raw, NOT normalized
    result: object          # TimelineResult, unsmoothed and ungated
    source: str             # "upload" | "scenario" | "test-example"
    truth: list = None      # ScenarioSegment list, scenario only
    snr_known: bool = False
    true_snr_db: float = None
    noise_power: float = 1.0
    thresholds: dict = field(default_factory=dict)
    smoothing_alpha: float = DEFAULT_ALPHA
    noise_gate: float = DEFAULT_NOISE_GATE
    hold_us: float = DEFAULT_HOLD_US
    # Which view the operator has selected. Lives on the session so every page
    # reads one source: RF Replay owns the toggle, but Overview and Alerts
    # read the same capture, and a console that reports 70 events on one page
    # and 7 on another -- with nothing on screen explaining the difference --
    # is worse than either number alone.
    display_smoothed: bool = True

    @property
    def duration_ms(self):
        return len(self.iq) / CFG["signal"]["fs"] * 1000.0

    def _resolved(self, smoothed):
        return smooth(self.result, self.smoothing_alpha) if smoothed else self.result

    def _rules(self, smoothed):
        """Display rules apply only in smoothed mode.

        Raw mode is meant to show what the model actually did, window by
        window, so a judge can see the jitter the operational view hides.
        Applying the gate and hold there would defeat that.
        """
        if smoothed:
            return {"noise_gate": self.noise_gate, "hold_us": self.hold_us}
        return {"noise_gate": None, "hold_us": 0.0}

    def events(self, smoothed=None):
        smoothed = self.display_smoothed if smoothed is None else smoothed
        return detections(self._resolved(smoothed), self.thresholds,
                           **self._rules(smoothed))

    def emitter_events(self, smoothed=None):
        """Events with an actual emitter -- empty channel is not an event."""
        return [e for e in self.events(smoothed) if e.classes != ("NOISE_FLOOR",)]

    def tiers(self, smoothed=None):
        smoothed = self.display_smoothed if smoothed is None else smoothed
        return tier_track(self._resolved(smoothed), self.thresholds,
                           **self._rules(smoothed))

    def judged_events(self, smoothed=None):
        """Events involving a judged class. NOISE_FLOOR can never appear --
        it is the absence of an emitter, so an alert on it would invert the
        purpose of both the Alerts page and the class."""
        judged = set(CFG["judged_classes"])
        return [e for e in self.events(smoothed) if judged & set(e.classes)]


def analyze(iq, model, source, hop=None, truth=None, true_snr_db=None,
            max_windows=MAX_WINDOWS):
    """Run inference over a capture and package the result.

    `truth` is accepted only for scenario captures; any other source has it
    forced to None, so a TRUTH overlay cannot be rendered over data we do not
    actually have ground truth for.
    """
    window_len = CFG["signal"]["window_len"]
    hop = hop or window_len

    n_windows = 1 + max(len(iq) - window_len, 0) // hop
    if n_windows > max_windows:
        raise ValueError(
            f"too many windows: {n_windows} at hop {hop} exceeds the "
            f"{max_windows} cap. Use a larger hop or a shorter capture."
        )

    result = classify_capture(iq, model, hop=hop)
    thresholds = dict(zip(CLASSES, resolve_multilabel_thresholds()))

    if source != "scenario":
        truth = None
        true_snr_db = None

    return CaptureSession(
        iq=np.asarray(iq), result=result, source=source, truth=truth,
        snr_known=true_snr_db is not None, true_snr_db=true_snr_db,
        noise_power=noise_floor_power(iq), thresholds=thresholds,
    )


def reanalyze(session, model, hop=None):
    """Re-run a DIFFERENT model over the capture already loaded.

    Without this, switching models meant clicking Synthesize again, which
    generates a fresh random capture -- so the two models were never compared
    on the same signal. Truth and known SNR carry over because they are
    properties of the capture, not of whichever model just looked at it.
    """
    fresh = analyze(session.iq, model, source=session.source,
                     hop=hop or session.result.hop, truth=session.truth,
                     true_snr_db=session.true_snr_db)
    fresh.display_smoothed = session.display_smoothed
    return fresh


def load_scenario(model, total_duration=0.05, hop=None, snr_db=0, seed=None,
                   case=None):
    """`case` names an entry in scenarios.CASES; None uses the default script.

    snr_db is per-emitter, referenced to the first non-jamming emitter, so the
    same value means the same thing whether the case has one emitter or three.
    """
    seed = np.random.randint(0, 100000) if seed is None else seed
    script = CASES.get(case) if case else None
    needs_library = script and any(c in CIVILIAN for c, _, _ in script)
    iq, segments = build_scenario(
        total_duration=total_duration, snr_db=snr_db, seed=seed, script=script,
        library=civilian_library() if needs_library else None)
    return analyze(iq, model, source="scenario", hop=hop, truth=segments,
                    true_snr_db=snr_db)


def load_upload(path, model, hop=None):
    """Interleaved float32 I,Q,I,Q,... -- same contract as src/infer.py.

    The whole file is analyzed. The old UI truncated to iq[:512] and silently
    discarded everything after the first 160 microseconds.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size < 2:
        raise ValueError(
            "File contains no complex samples. Expected interleaved float32 "
            "I,Q,I,Q,... -- at least 2 values."
        )
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw[0::2] + 1j * raw[1::2]
    return analyze(iq, model, source="upload", hop=hop)


def load_test_example(model, X, y, snr_labels, idx):
    """One held-out example: exactly 512 samples, so exactly one window and no
    timeline. Pages must render correctly in that degenerate case."""
    arr = X[idx]
    iq = arr[0] + 1j * arr[1]
    session = analyze(iq, model, source="test-example",
                       hop=CFG["signal"]["window_len"])
    session.snr_known = True
    session.true_snr_db = float(snr_labels[idx])
    return session
