"""
Label rules for the week-1 60-class dataset: windows whose honest answer is
more than one class, even with a single generator behind them.

Three rules, decided 2026-11-04 from the seed-1 confusions:

  DRFM      A JAM_DRFM window holds the victim radar's own pulse AND the
            jammer's delayed, Doppler-shifted copy (src/generators/jam.py
            _drfm). Two emitters are on the air, so the window is labelled
            JAM_DRFM + the radar class it copied. Labelled JAM_DRFM alone, it
            taught the model that a plain radar pulse is sometimes "not radar":
            JAM_DRFM was the top confusion of RADAR_BARKER (23%), RADAR_PULSE
            (15%) and RADAR_POLY_BARKER (13%).

  AMBIGUOUS Classes no 160 us window can tell apart (configs week1.
            ambiguous_labels). RADAR_CW and JAM_TONE are both one unmodulated
            tone: they differ only in a 0-20 kHz starting offset that the
            impairment module's random placement erases (RADAR_CW was right
            3% of the time, 31% called JAM_TONE). A window carrying any class
            of a group carries all of them; telling radar CW from a tone
            jammer is left to context (the tracker: duration, power, place).

  SWEEP     LORA_CSS and JAM_SWEPT, window by window. A LoRa window cut inside
            one long chirp, and a slow jammer sweep at a LoRa chirp's rate,
            are the same straight sweep (src/generators/spread.py
            looks_like_lora_sweep; generators report it as lora_like_sweep).
            Those windows carry both labels; a window showing a LoRa wrap or
            turn, or a jammer sweep at a non-LoRa rate, keeps its own. The
            tracker settles it: LoRa wraps again within one symbol (<= 512 us).

New shards get these labels from the shard scripts (the generators report
what they need). Shards made before that carry one label; scripts/
merge_shards.py fixes them with fix_shard_labels, which replays each window
from its stored seed -- and refuses to guess: the replay must reproduce the
stored window bit for bit on a sample of rows, or the merge stops.

standalone_pairs is the one definition of "a single emitter" for scoring, so
training, the report and the fair check stay in step: one label, a DRFM pair,
an ambiguous group or the sweep pair.
"""
import numpy as np

from src.config import CFG, DATASET_CLASS_TO_IDX, DATASET_CLASSES

LABEL_RULES = "labels-v2"          # goes into the merge's combined_version salt
DRFM = "JAM_DRFM"
SWEEP_PAIR = ("LORA_CSS", "JAM_SWEPT")
AMBIGUOUS_GROUPS = tuple(tuple(g) for g in CFG.get("week1", {}).get("ambiguous_labels", []))

_DRFM_IDX = DATASET_CLASS_TO_IDX[DRFM]
_GROUP_IDX = [np.array([DATASET_CLASS_TO_IDX[c] for c in g]) for g in AMBIGUOUS_GROUPS]
# Label sets scored as one merged result (every member gets the window).
_SCORED_TOGETHER = _GROUP_IDX + [np.array([DATASET_CLASS_TO_IDX[c] for c in SWEEP_PAIR])]
REPLAY_CHECK_ROWS = 8              # rows per class and shard re-generated in full to prove the replay


def _radar():
    from src.generators import radar          # imported late: radar imports src.data
    return radar


def _module(cls):
    """Generator module that makes `cls` -- for the classes with a per-window rule."""
    from src.generators import jam, spread
    return spread if cls in spread.CLASSES else jam


# Classes whose labels depend on the generated window (replayed for old shards).
PER_WINDOW = (DRFM,) + SWEEP_PAIR


def _param(params, key, cls):
    v = np.nan if params is None else params.get(key, np.nan)
    if not np.isfinite(v):
        raise ValueError(f"a {cls} window needs its generator params ({key})")
    return v


def window_labels(cls, params=None, ambiguous=True):
    """Every label for one generated window of class `cls` with generator
    `params`: the class, the radar a DRFM window copied, the other class of
    the sweep pair for a straight LoRa-rate sweep, and (ambiguous=True) the
    rest of its ambiguous group. Shard scripts pass ambiguous=False: the
    groups are a merge-time rule, so changing them never needs new shards."""
    names = {cls}
    if cls == DRFM:
        names.add(_radar().CLASSES[int(_param(params, "copied_radar", cls))])
    if cls in SWEEP_PAIR and _param(params, "lora_like_sweep", cls) == 1:
        names |= set(SWEEP_PAIR)
    if ambiguous:
        for group in AMBIGUOUS_GROUPS:
            if names & set(group):
                names |= set(group)
    return names


def apply_ambiguous(y):
    """y (N, C) 0/1 -> copy where every row that has one class of an
    ambiguous group has all of them."""
    y = np.array(y, copy=True)
    for idx in _GROUP_IDX:
        y[:, idx] = y[:, idx].max(axis=1, keepdims=True)
    return y


def replayed_labels(cls, seed):
    """window_labels for a generated window of class `cls` from its stored
    seed. The generator's draws come before any impairment, so the clean
    replay reports the same params as the stored (impaired) window had."""
    _, params = _module(cls).generate_window(cls, None, np.random.default_rng(int(seed)))
    return window_labels(cls, params, ambiguous=False)


def drfm_copied_class(seed):
    """The radar class a generated JAM_DRFM window copied, from its stored seed."""
    return next(iter(replayed_labels(DRFM, seed) - {DRFM}))


def _check_replay(cls, data, rows, impaired):
    from src.data import impairments
    mod = _module(cls)
    for r in rows:
        rng = np.random.default_rng(int(data["seed"][r]))
        snr = float(data["snr_db"][r])
        if impaired:
            again, _ = impairments.generate_impaired(mod, cls, snr, rng)
        else:
            again, _ = mod.generate_window(cls, snr, rng)
        if not np.array_equal(again, data["iq"][r]):
            raise RuntimeError(
                f"{cls} window with seed {int(data['seed'][r])} does not regenerate bit for bit, so its "
                f"labels cannot be recovered from its seed. The generator changed since this shard was "
                f"made: regenerate this shard ({mod.__name__}).")


def fix_shard_labels(data, manifest=None):
    """Labels for one loaded shard with every rule applied: (y uint8 (N, C),
    {class: rows that gained a label by replay}).

    Generated JAM_DRFM, JAM_SWEPT and LORA_CSS rows that still carry their
    class alone (shards made before the rules) are replayed from their seeds;
    rows that already carry more (newer shards) are left as they are."""
    y = np.array(data["y"], copy=True)
    gen = np.asarray(data["source"]) == "gen"
    impaired = (manifest or {}).get("notes", {}).get("impairments", True)
    fixed = {}
    for cls in PER_WINDOW:
        # Re-counted per class: a LoRa row that just gained JAM_SWEPT is done.
        single = y.sum(axis=1) == 1
        todo = np.flatnonzero(single & gen & (y[:, DATASET_CLASS_TO_IDX[cls]] == 1))
        if not len(todo):
            continue
        _check_replay(cls, data, todo[:REPLAY_CHECK_ROWS], impaired)
        for r in todo:
            for name in replayed_labels(cls, data["seed"][r]):
                y[r, DATASET_CLASS_TO_IDX[name]] = 1
        changed = int((y[todo].sum(axis=1) > 1).sum())
        if changed:
            fixed[cls] = changed
    return apply_ambiguous(y), fixed


def generated_class(y, group):
    """Class index each row of a shard (labels as written by the shard
    scripts, before fix_shard_labels) was generated as, or -1 for a row that
    is not one emitter (a mixture). A DRFM row is JAM_DRFM; a sweep-pair row
    is LORA_CSS in the spread group, JAM_SWEPT elsewhere."""
    y = np.asarray(y).astype(bool)
    k = y.sum(axis=1)
    out = np.where(k == 1, y.argmax(axis=1), -1)
    out[(k == 2) & y[:, _DRFM_IDX]] = _DRFM_IDX
    pair = _SCORED_TOGETHER[-1]
    is_pair = (k == 2) & y[:, pair].all(axis=1)
    out[is_pair] = DATASET_CLASS_TO_IDX[SWEEP_PAIR[0] if group == "spread" else SWEEP_PAIR[1]]
    return out


def standalone_pairs(y):
    """(rows, classes): every single-emitter window paired with the class it
    is scored under. One label -> that class; JAM_DRFM + one radar class ->
    JAM_DRFM; exactly an ambiguous group or the sweep pair -> every member
    (scored as one merged result). Mixtures -> no pair."""
    y = np.asarray(y).astype(bool)
    k = y.sum(axis=1)
    rows, classes = [np.flatnonzero(k == 1)], [y[k == 1].argmax(axis=1)]
    radar_idx = np.array([DATASET_CLASS_TO_IDX[c] for c in _radar().CLASSES])
    drfm = np.flatnonzero((k == 2) & y[:, _DRFM_IDX] & (y[:, radar_idx].sum(axis=1) == 1))
    rows.append(drfm)
    classes.append(np.full(len(drfm), _DRFM_IDX))
    for idx in _SCORED_TOGETHER:
        g = np.flatnonzero((k == len(idx)) & y[:, idx].all(axis=1))
        for c in idx:
            rows.append(g)
            classes.append(np.full(len(g), c))
    return np.concatenate(rows).astype(int), np.concatenate(classes).astype(int)


def scored_prediction(classes, top, y_rows):
    """Top-1 predictions for a confusion matrix: a prediction that is one of
    the window's own labels (y_rows, one row per pair) counts as the class
    it is scored under, so a merged pair never shows up as each other's
    mistake -- but a LoRa window that shows its wrap still does."""
    top = np.asarray(top).copy()
    ok = np.asarray(y_rows).astype(bool)[np.arange(len(top)), top]
    top[ok] = np.asarray(classes)[ok]
    return top


def class_names(row):
    """Class names set in one multi-hot row."""
    return {DATASET_CLASSES[i] for i in np.flatnonzero(np.asarray(row))}
