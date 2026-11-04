"""
Label rules for the week-1 60-class dataset: windows whose honest answer is
more than one class, even with a single generator behind them.

Two rules, both decided 2026-11-04 from the seed-1 confusions:

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

New jam shards get the DRFM label from make_jam_shards.py (the generator
reports the copied class). Shards made before that carry JAM_DRFM alone;
scripts/merge_shards.py fixes them with fix_shard_labels, which recovers the
copied class by replaying each window from its stored seed -- and refuses to
guess: the replay must reproduce the stored window bit for bit on a sample of
rows, or the merge stops.

standalone_pairs is the one definition of "a single emitter" for scoring, so
training, the report and the fair check stay in step: one label, a DRFM pair,
or a whole ambiguous group.
"""
import numpy as np

from src.config import CFG, DATASET_CLASS_TO_IDX, DATASET_CLASSES

LABEL_RULES = "labels-v1"          # goes into the merge's combined_version salt
DRFM = "JAM_DRFM"
AMBIGUOUS_GROUPS = tuple(tuple(g) for g in CFG.get("week1", {}).get("ambiguous_labels", []))

_DRFM_IDX = DATASET_CLASS_TO_IDX[DRFM]
_GROUP_IDX = [np.array([DATASET_CLASS_TO_IDX[c] for c in g]) for g in AMBIGUOUS_GROUPS]
_MATE = {DATASET_CLASS_TO_IDX[c]: set(idx.tolist()) for g, idx in zip(AMBIGUOUS_GROUPS, _GROUP_IDX)
         for c in g}
REPLAY_CHECK_ROWS = 8              # DRFM rows per shard re-generated in full to prove the replay


def _radar():
    from src.generators import radar          # imported late: radar imports src.data
    return radar


def window_labels(cls, params=None, ambiguous=True):
    """Every label for one generated window of class `cls` with generator
    `params`: the class, the radar a DRFM window copied, and (ambiguous=True)
    the rest of its ambiguous group. Shard scripts pass ambiguous=False: the
    groups are a merge-time rule, so changing them never needs new shards."""
    names = {cls}
    if cls == DRFM:
        if params is None or not np.isfinite(params.get("copied_radar", np.nan)):
            raise ValueError("a JAM_DRFM window needs its generator params (copied_radar)")
        names.add(_radar().CLASSES[int(params["copied_radar"])])
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


def drfm_copied_class(seed):
    """The radar class a generated JAM_DRFM window copied, from its stored
    seed. The generator's first draws pick the copied pulse, and they happen
    before any impairment, so the clean replay names the same class."""
    from src.generators import jam
    _, params = jam.generate_window(DRFM, None, np.random.default_rng(int(seed)))
    return _radar().CLASSES[int(params["copied_radar"])]


def _check_replay(data, rows, impaired):
    from src.data import impairments
    from src.generators import jam
    for r in rows:
        rng = np.random.default_rng(int(data["seed"][r]))
        snr = float(data["snr_db"][r])
        if impaired:
            again, _ = impairments.generate_impaired(jam, DRFM, snr, rng)
        else:
            again, _ = jam.generate_window(DRFM, snr, rng)
        if not np.array_equal(again, data["iq"][r]):
            raise RuntimeError(
                f"JAM_DRFM window with seed {int(data['seed'][r])} does not regenerate bit for bit, "
                "so the copied radar cannot be recovered from its seed. The jam generator changed "
                "since this shard was made: regenerate the jam shards (scripts/make_jam_shards.py).")


def fix_shard_labels(data, manifest=None):
    """Labels for one loaded shard with both rules applied: (y uint8 (N, C),
    number of DRFM rows that gained their radar label).

    Only generated JAM_DRFM rows still labelled JAM_DRFM alone are replayed;
    rows that already carry a radar label (newer shards) are left as they are."""
    y = np.array(data["y"], copy=True)
    radar_idx = np.array([DATASET_CLASS_TO_IDX[c] for c in _radar().CLASSES])
    todo = np.flatnonzero((y[:, _DRFM_IDX] == 1) & (y[:, radar_idx].sum(axis=1) == 0)
                          & (np.asarray(data["source"]) == "gen"))
    if len(todo):
        impaired = (manifest or {}).get("notes", {}).get("impairments", True)
        _check_replay(data, todo[:REPLAY_CHECK_ROWS], impaired)
        for r in todo:
            y[r, DATASET_CLASS_TO_IDX[drfm_copied_class(data["seed"][r])]] = 1
    return apply_ambiguous(y), len(todo)


def standalone_pairs(y):
    """(rows, classes): every single-emitter window paired with the class it
    is scored under. One label -> that class; JAM_DRFM + one radar class ->
    JAM_DRFM; exactly one ambiguous group -> every member (so RADAR_CW and
    JAM_TONE are scored as one merged result). Mixtures -> no pair."""
    y = np.asarray(y).astype(bool)
    k = y.sum(axis=1)
    rows, classes = [np.flatnonzero(k == 1)], [y[k == 1].argmax(axis=1)]
    radar_idx = np.array([DATASET_CLASS_TO_IDX[c] for c in _radar().CLASSES])
    drfm = np.flatnonzero((k == 2) & y[:, _DRFM_IDX] & (y[:, radar_idx].sum(axis=1) == 1))
    rows.append(drfm)
    classes.append(np.full(len(drfm), _DRFM_IDX))
    for idx in _GROUP_IDX:
        g = np.flatnonzero((k == len(idx)) & y[:, idx].all(axis=1))
        for c in idx:
            rows.append(g)
            classes.append(np.full(len(g), c))
    return np.concatenate(rows).astype(int), np.concatenate(classes).astype(int)


def scored_prediction(classes, top):
    """Top-1 predictions for a confusion matrix: a prediction inside the true
    class's ambiguous group counts as that class, so the merged pair does not
    show up as each other's mistake."""
    top = np.asarray(top).copy()
    for i, (c, t) in enumerate(zip(classes, top)):
        if t != c and t in _MATE.get(int(c), ()):
            top[i] = c
    return top


def class_names(row):
    """Class names set in one multi-hot row."""
    return {DATASET_CLASSES[i] for i in np.flatnonzero(np.asarray(row))}
