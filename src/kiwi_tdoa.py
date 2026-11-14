"""
Where is a shortwave transmitter? Time difference of arrival (TDoA) from a
few GPS-timed public KiwiSDRs, computed on this machine.

KiwiSDR's own TDoA service (the TDoA extension on every GPS KiwiSDR) runs on
tdoa.kiwisdr.com and needs an access string each KiwiSDR gives only to its own
web page, so NEXA does not call it. Instead it follows the same method, ported
from the open-source Octave code (github.com/hcab14/TDoA) to numpy:

  1. kiwirecorder records the same frequency on 3 (at most 4) KiwiSDRs at once
     as IQ with GPS time stamps (--kiwi-wav: a 'kiwi' chunk before every block
     of samples). read_kiwi_wav() turns that into samples + GPS time per sample.
  2. pair_lag() cross-correlates each pair in 1 s blocks: the peak says how much
     later the signal reached one receiver than the other (arrival-time
     difference). Outliers are dropped; the mean and spread are kept.
  3. locate() tries every point of a lat/lon grid: the arrival-time differences
     a transmitter there would give (great-circle distance / speed of light)
     against the measured ones. The best point is "most likely"; the points
     still consistent with the data (95 %) are the area.

Short-wave signals reach the receivers via the ionosphere, on paths longer than
the ground distance, so the answer is an area tens to hundreds of km across,
not a point. Pure functions only (no network); scripts/sdr_hub.py runs the job.
"""
import math
import re
import struct

import numpy as np

C_KM_S = 299792.458
EARTH_KM = 6371.0
CHI2_95_2DOF = 5.99          # h - h_min below this: inside the 95 % area
SKYWAVE_SIGMA_S = 300e-6     # added to every pair: ionospheric paths differ from the ground distance
                             # (~90 km of path the measurement cannot see). The first live run
                             # (15295 kHz, 2026-11-14, DU6/PE1NSQ + JKT + HS0ZQS) agreed with itself to
                             # 2 us yet sat 290 km from the Kajang transmitter: the error is systematic.


# ---------------------------------------------------------------- receivers

def parse_status(text):
    """A KiwiSDR /status page (key=value lines) -> dict with what TDoA needs:
    lat, lon (None if the Kiwi gives no position), gps_good (satellites in the
    solution), fixes_min (GPS fixes in the last minute), users, users_max, name,
    tdoa_id, tdoa_ch (channels it lets TDoA use; 0 = none)."""
    kv = {}
    for line in str(text or "").splitlines():
        k, sep, v = line.partition("=")
        if sep:
            kv[k.strip()] = v.strip()

    def num(k, cast=int):
        try:
            return cast(kv[k])
        except (KeyError, ValueError):
            return None
    lat = lon = None
    m = re.fullmatch(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", kv.get("gps", ""))
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (abs(lat) <= 90 and abs(lon) <= 180) or (lat == 0 and lon == 0):
            lat = lon = None
    return {"lat": lat, "lon": lon, "gps_good": num("gps_good"), "fixes_min": num("fixes_min"),
            "users": num("users"), "users_max": num("users_max"), "name": kv.get("name", ""),
            "loc": kv.get("loc", ""),
            "tdoa_id": kv.get("tdoa_id") or None, "tdoa_ch": num("tdoa_ch"),
            "offline": kv.get("offline") == "yes", "status": kv.get("status", "")}


def status_problem(st):
    """None if this KiwiSDR can take part in a TDoA run now, else why not (in words)."""
    if st.get("offline") or st.get("status") not in ("", "active"):
        return "offline"
    if st.get("lat") is None:
        return "gives no GPS position"
    if not (st.get("fixes_min") or 0) > 0 and not (st.get("gps_good") or 0) >= 3:
        return "has no GPS fix now (its time stamps would be wrong)"
    if st.get("tdoa_ch") == 0:
        return "does not allow TDoA"
    if st.get("users") is not None and st.get("users_max") and st["users"] >= st["users_max"]:
        return "is full (no free channel)"
    return None


# ---------------------------------------------------------------- GPS-stamped IQ

def read_kiwi_wav(path, max_last_fix=254):
    """A kiwirecorder --kiwi-wav IQ file -> (t, z, fs): GPS time of every sample
    (seconds, float64), complex64 samples, sample rate. Raises ValueError for a
    file without GPS time stamps.

    Like hcab14's proc_kiwi_iq_wav.m: the first block is dropped, blocks whose
    GPS solution is too old (last_gps_solution >= max_last_fix; 255 = never) are
    dropped, and the time of each sample is interpolated between the blocks
    that carry a fresh GPS time stamp (others are the Kiwi's extrapolation)."""
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fs_hdr, blocks = 12, None, []
    stamp = None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
        if cid == b"fmt ":
            if len(body) < 8:
                raise ValueError(f"{path}: broken fmt chunk")
            tag, nch, fs_hdr = struct.unpack("<HHI", body[:8])
            if tag != 1 or nch != 2:
                raise ValueError(f"{path}: not a 2-channel IQ WAV")
        elif cid == b"kiwi":
            if len(body) < 10:                         # cut off at the end of the recording
                break
            last, _, sec, nsec = struct.unpack("<BBII", body[:10])
            stamp = (last, sec + 1e-9 * nsec)
        elif cid == b"data":
            if stamp is None:
                raise ValueError(f"{path}: no KiwiSDR GPS time stamps (record with --kiwi-wav)")
            n = len(body) // 4
            iq = np.frombuffer(body[:n * 4], "<i2").astype(np.float32) / 32768
            blocks.append((stamp[0], stamp[1], (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)))
            stamp = None
    blocks = [b for b in blocks[1:] if b[0] < max_last_fix and len(b[2])]
    if len(blocks) < 3:
        raise ValueError(f"{path}: too few GPS-timed blocks ({len(blocks)})")
    last = np.array([b[0] for b in blocks], float)
    ts = np.array([b[1] for b in blocks], float)
    start = np.concatenate([[0], np.cumsum([len(b[2]) for b in blocks])[:-1]]).astype(float)
    z = np.concatenate([b[2] for b in blocks])
    fresh = np.flatnonzero(np.diff(last) < 0) + 1       # the solution age dropped: a new GPS fix
    if last[0] == 0:
        fresh = np.concatenate([[0], fresh])
    if len(fresh) < 2:                                  # no fresh fixes seen: trust every block
        fresh = np.arange(len(blocks))
    fs_all = np.diff(start[fresh]) / np.diff(ts[fresh])
    fs = float(np.median(fs_all[np.isfinite(fs_all) & (fs_all > 0)])) if len(fs_all) else float(fs_hdr or 12000)
    resid = ts[fresh] - (ts[fresh][0] + (start[fresh] - start[fresh][0]) / fs)
    good = np.abs(resid - np.median(resid)) * fs <= 1.0  # more than one sample off: a bad stamp
    fresh = fresh[good] if good.sum() >= 2 else fresh
    idx = np.arange(len(z), dtype=float)
    t = np.interp(idx, start[fresh], ts[fresh])
    lo, hi = idx < start[fresh][0], idx > start[fresh][-1]
    t[lo] = ts[fresh][0] - (start[fresh][0] - idx[lo]) / fs        # extrapolate at the ends
    t[hi] = ts[fresh][-1] + (idx[hi] - start[fresh][-1]) / fs
    return t, z, fs


def write_kiwi_wav(path, t0, z, fs, block=512, last_fix=0):
    """The inverse of read_kiwi_wav, for tests: complex samples starting at GPS
    time t0, in blocks of `block` samples each with a 'kiwi' time stamp."""
    chunks = [struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, int(round(fs)), int(round(fs)) * 4, 4, 16)]
    for k in range(0, len(z) - block + 1, block):
        t = t0 + k / fs
        sec = int(t)
        chunks.append(struct.pack("<4sIBBII", b"kiwi", 10, last_fix if k % (block * 8) == 0 else 1, 0, sec,
                                  int(round((t - sec) * 1e9))))
        iq = np.empty(2 * block, "<i2")
        iq[0::2] = np.clip(np.round(z[k:k + block].real * 32767), -32768, 32767)
        iq[1::2] = np.clip(np.round(z[k:k + block].imag * 32767), -32768, 32767)
        chunks.append(struct.pack("<4sI", b"data", iq.nbytes) + iq.tobytes())
    body = b"WAVE" + b"".join(chunks)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


# ---------------------------------------------------------------- arrival-time differences

CLUSTER_TOL_S = 100e-6       # blocks whose lags are this close count as the same arrival
MIN_AGREE = 0.4              # at least this share of blocks (and 3) must agree on one lag
PATH_SHARE = 0.15            # a second lag this many blocks agree on is a second ionospheric path
REFINE_STEPS = 32            # sub-sample positions tried round the peak


def _refine(c, m, fs, half=8):
    """Sub-sample position of the correlation peak near index m: the complex
    correlation is band-limited, so sinc interpolation between samples is exact
    (a 3-point parabola on |c| of a broad AM peak is pulled toward whole samples,
    which made lags jump by one sample, ~83 us, from block to block).
    -> (offset in samples from m, peak height)"""
    lo, hi = max(0, m - half), min(len(c), m + half + 1)
    k = np.arange(lo, hi) - m
    d = np.linspace(-1.0, 1.0, 2 * REFINE_STEPS + 1)
    vals = np.abs(np.sinc(d[:, None] - k[None, :]) @ c[lo:hi])
    best = int(np.argmax(vals))
    return float(d[best]), float(vals[best])


def pair_lag(a, b, max_dt_s=0.02, block_s=1.0, min_peak=0.1, far=None):
    """Arrival-time difference of one signal at receivers a and b (each (t, z, fs)
    from read_kiwi_wav): how much LATER it reached a than b, in seconds.

    -> {"lag_s", "sigma_s", "blocks", "used", "agree", "peak"} or None. Per 1 s block
    (as hcab14's tdoa_compute_lags.m) the highest cross-correlation peak within
    +-max_dt_s, placed to a fraction of a sample. Then the lag that the most blocks
    agree on (within CLUSTER_TOL_S) is kept, weighted by peak height squared;
    None if fewer than MIN_AGREE of the blocks (or 3) agree. On a broadcast the
    second-best peak is often almost as high as the best, so one block alone
    proves little; agreement across blocks does.
    far ("a" or "b"): the receiver further from the others. When the blocks split
    between two lags (a far receiver hearing one and two ionospheric hops, ~1 ms
    apart; 15295 kHz in Taiwan on 14 Nov), the one where the far receiver hears the
    signal EARLIEST is kept: the extra hop only ever adds path."""
    (t1, z1, fs1), (t2, z2, fs2) = a, b
    if abs(fs1 - fs2) > 0.01 * fs1:
        raise ValueError(f"sample rates differ ({fs1:.0f} vs {fs2:.0f} S/s)")
    fs = (fs1 + fs2) / 2
    B, M = int(block_s * fs), int(math.ceil(max_dt_s * fs)) + 10
    T0, T1 = max(t1[0], t2[0]) + M / fs, min(t1[-1], t2[-1]) - (B + M) / fs
    lags, peaks = [], []
    blocks = 0
    for T in np.arange(T0, T1, block_s):
        i, j = int(np.searchsorted(t1, T)), int(np.searchsorted(t2, T))
        if i - M < 0 or i + B + M > len(z1) or j + B > len(z2):
            continue
        blocks += 1
        x1 = z1[i - M:i + B + M] - z1[i - M:i + B + M].mean()   # without the carrier
        x2 = z2[j:j + B] - z2[j:j + B].mean()
        c = np.correlate(x1, x2, "valid")                          # c[m] = sum x1[m+k] conj(x2[k])
        norm = math.sqrt(float(np.vdot(x1[M:M + B], x1[M:M + B]).real * np.vdot(x2, x2).real)) or 1.0
        c = c / norm
        base = t1[i] - t2[j]                                       # lag of index m is base + (m - M)/fs
        lag_t = base + (np.arange(len(c)) - M) / fs
        ok = np.flatnonzero(np.abs(lag_t) <= max_dt_s)
        if len(ok) < 3:
            continue
        m = int(ok[np.argmax(np.abs(c[ok]))])
        if abs(c[m]) < min_peak:
            continue
        frac, height = _refine(c, m, fs)
        lags.append(base + (m - M + frac) / fs)
        peaks.append(height)
    if not lags:
        return None
    lags, peaks = np.array(lags), np.array(peaks)
    groups, left = [], np.ones(len(lags), bool)                     # lags most blocks agree on, best first
    while left.any():
        idx = np.flatnonzero(left)
        votes = [np.sum(np.abs(lags[idx] - lags[k]) <= CLUSTER_TOL_S) + 1e-3 * np.sum(
            peaks[idx][np.abs(lags[idx] - lags[k]) <= CLUSTER_TOL_S] ** 2) for k in idx]
        centre = lags[idx[int(np.argmax(votes))]]
        member = left & (np.abs(lags - centre) <= CLUSTER_TOL_S)
        groups.append(member)
        left &= ~member
    keep = groups[0]
    paths = [g for g in groups if g.sum() >= max(3, PATH_SHARE * blocks)]
    if far in ("a", "b") and len(paths) > 1:                        # two paths: the earliest at the far receiver
        mean = [float(np.mean(lags[g])) for g in paths]
        keep = paths[int(np.argmax(mean) if far == "b" else np.argmin(mean))]   # lag = t_a - t_b
    if keep.sum() < max(3, MIN_AGREE * blocks) and not (len(paths) > 1 and keep.sum() >= 3):
        return None                                                # no lag most blocks agree on: no answer
    w = peaks[keep] ** 2
    lag = float(np.sum(w * lags[keep]) / np.sum(w))
    var = float(np.sum(w * (lags[keep] - lag) ** 2) / np.sum(w))
    sigma = math.sqrt(max(var, (0.1 / fs) ** 2))                   # never sharper than a tenth of a sample
    return {"lag_s": lag, "sigma_s": sigma, "blocks": blocks, "used": int(keep.sum()),
            "agree": round(float(keep.sum()) / blocks, 2), "peak": round(float(np.median(peaks[keep])), 3),
            "paths": len(paths)}


# ---------------------------------------------------------------- where

def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_KM * np.arccos(np.clip(c, -1.0, 1.0))


def _h(stations, pairs, lat, lon):
    dt = [great_circle_km(s["lat"], s["lon"], lat, lon) / C_KM_S for s in stations]
    h = np.zeros(np.broadcast(lat, lon).shape)
    for p in pairs:
        h += (dt[p["i"]] - dt[p["j"]] - p["lag_s"]) ** 2 / p["sigma_s"] ** 2
    return h


def locate(stations, pairs, margin_deg=20.0, step_deg=0.1):
    """stations: [{"lat", "lon", ...}], pairs: [{"i", "j", "lag_s", "sigma_s"}]
    (lag_s = arrival at station i minus arrival at station j). Grid search over the
    receivers' region (+-margin_deg), then a finer grid round the best point.

    -> {"lat", "lon", "radius_km", "area": [lat_s, lat_n, lon_w, lon_e], "fit"}:
    the most likely point, the 95 % area round it (as a box and the radius of a
    circle of the same size) and how well the best point fits (h_min; 0 = perfect)."""
    if len(stations) < 3 or len(pairs) < 2:
        raise ValueError("TDoA needs 3 receivers: with 2 the answer is a curve, not a place")
    lats = [s["lat"] for s in stations]
    lons = [s["lon"] for s in stations]
    lat_ax = np.arange(max(min(lats) - margin_deg, -89.0), min(max(lats) + margin_deg, 89.0) + 1e-9, step_deg)
    lon_ax = np.arange(min(lons) - margin_deg, max(lons) + margin_deg + 1e-9, step_deg)
    la, lo = np.meshgrid(lat_ax, lon_ax, indexing="ij")
    h = _h(stations, pairs, la, lo)
    k = np.unravel_index(np.argmin(h), h.shape)
    fine = step_deg / 10                                   # refine round the best point
    lat_f = np.arange(lat_ax[k[0]] - 2 * step_deg, lat_ax[k[0]] + 2 * step_deg + 1e-9, fine)
    lon_f = np.arange(lon_ax[k[1]] - 2 * step_deg, lon_ax[k[1]] + 2 * step_deg + 1e-9, fine)
    lf, of = np.meshgrid(lat_f, lon_f, indexing="ij")
    hf = _h(stations, pairs, lf, of)
    kf = np.unravel_index(np.argmin(hf), hf.shape)
    best_lat, best_lon, h_min = float(lat_f[kf[0]]), float(lon_f[kf[1]]), float(hf[kf])
    inside = h <= min(h_min, float(h[k])) + CHI2_95_2DOF   # the 95 % area, on the coarse grid
    cell_km2 = (step_deg * 111.2) ** 2 * np.cos(np.radians(la))
    area_km2 = float(np.sum(cell_km2[inside])) or (step_deg * 111.2) ** 2
    ii, jj = np.nonzero(inside)
    box = ([float(lat_ax[ii.min()]), float(lat_ax[ii.max()]), float(lon_ax[jj.min()]), float(lon_ax[jj.max()])]
           if len(ii) else [best_lat, best_lat, best_lon, best_lon])
    return {"lat": round(best_lat, 3), "lon": round(best_lon, 3), "radius_km": round(math.sqrt(area_km2 / math.pi)),
            "area": [round(v, 2) for v in box], "fit": round(h_min, 2)}


def solve(recordings, max_dt_extra_s=0.003):
    """recordings: [{"id", "lat", "lon", "t", "z", "fs"}] (3 to 5 receivers).
    -> {"lat", "lon", "radius_km", "area", "fit", "pairs": [...], "stations": [...]}.
    Raises ValueError when fewer than 3 receivers gave a usable correlation."""
    return combine(*measure(recordings, max_dt_extra_s))


def to_rate(t, z, fs, target):
    """One recording brought down to the sample rate `target` (KiwiSDRs differ: most
    give 12 kS/s IQ, some 6 kS/s), filtered against aliasing, with the GPS time of
    every new sample interpolated from the old ones. -> (t, z, fs)"""
    if fs <= target * 1.01:
        return t, z, fs
    from fractions import Fraction
    from scipy.signal import resample_poly
    r = Fraction(target / fs).limit_denominator(64)
    y = resample_poly(z, r.numerator, r.denominator).astype(np.complex64)
    pos = np.arange(len(y)) * (r.denominator / r.numerator)          # where each new sample sits, in old samples
    keep = pos <= len(z) - 1
    return np.interp(pos[keep], np.arange(len(z)), t), y[keep], fs * r.numerator / r.denominator


def measure(recordings, max_dt_extra_s=0.003):
    """The arrival-time difference of every pair of one recording set. Receivers
    with different sample rates are first brought to the lowest one; a pair that
    still cannot be compared is left out, never the whole run.
    -> (stations [{"id", "lat", "lon"}], pairs [{"i", "j", "a", "b", "lag_s", ...}])"""
    low = min(r["fs"] for r in recordings)
    recordings = [dict(r, **dict(zip(("t", "z", "fs"), to_rate(r["t"], r["z"], r["fs"], low)))) for r in recordings]
    pairs = []
    # how far each receiver is from the others (mean distance): the further one of a
    # pair is the one more likely to hear an extra ionospheric hop
    spread = [np.mean([float(great_circle_km(r["lat"], r["lon"], q["lat"], q["lon"])) for q in recordings if q is not r])
              for r in recordings]
    for i in range(len(recordings)):
        for j in range(i + 1, len(recordings)):
            a, b = recordings[i], recordings[j]
            base_km = float(great_circle_km(a["lat"], a["lon"], b["lat"], b["lon"]))
            try:
                r = pair_lag((a["t"], a["z"], a["fs"]), (b["t"], b["z"], b["fs"]),
                             max_dt_s=base_km / C_KM_S + max_dt_extra_s, far="a" if spread[i] > spread[j] else "b")
            except ValueError:                                   # e.g. rates still apart: skip this pair only
                r = None
            if r:
                pairs.append({"i": i, "j": j, "a": a["id"], "b": b["id"], **r,
                              "far": "a" if spread[i] > spread[j] else "b",
                              "lag_us": round(r["lag_s"] * 1e6, 1), "sigma_us": round(r["sigma_s"] * 1e6, 1)})
    stations = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in recordings]
    return stations, pairs


def merge_runs(runs):
    """Several measure() results of the same frequency (one per run) -> one
    (stations, pairs): each pair's lag is the median over the runs that have it,
    its spread the larger of the runs' own and their disagreement. A wrong peak in
    one run is outvoted by the others. But when the runs split into two lags a
    sky-wave hop apart (a far receiver on one hop in some runs, two in others:
    Taiwan on 14 Nov), the one where the far receiver heard it EARLIEST is kept,
    as within one run: any run that caught the direct path wins."""
    stations, index, by_pair = [], {}, {}
    for st, ps in runs:
        for s in st:
            if s["id"] not in index:
                index[s["id"]] = len(stations)
                stations.append(dict(s))
        for p in ps:
            a, b, lag, far = p["a"], p["b"], p["lag_s"], p.get("far")
            if index[a] > index[b]:
                a, b, lag, far = b, a, -lag, {"a": "b", "b": "a"}.get(far)
            by_pair.setdefault((a, b), []).append({**p, "lag_s": lag, "far": far})
    pairs = []
    for (a, b), got in by_pair.items():
        lags = np.array([g["lag_s"] for g in got])
        far = next((g["far"] for g in got if g.get("far")), None)
        two = False
        if far and np.ptp(lags) > RUN_HOP_S:                       # runs disagree by a hop: earliest at the far one
            pick = lags.max() if far == "b" else lags.min()        # lag = t_a - t_b
            near = np.abs(lags - pick) <= RUN_HOP_S / 2
            lags, got, two = lags[near], [g for g, n in zip(got, near) if n], True
        lag = float(np.median(lags))
        mad = 1.4826 * float(np.median(np.abs(lags - lag))) if len(lags) > 2 else float(np.ptp(lags)) / 2
        sigma = max(float(np.median([g["sigma_s"] for g in got])), mad)
        pairs.append({"i": index[a], "j": index[b], "a": a, "b": b, "lag_s": lag, "sigma_s": sigma,
                      "lag_us": round(lag * 1e6, 1), "sigma_us": round(sigma * 1e6, 1), "runs": len(got),
                      "blocks": sum(g.get("blocks", 0) for g in got), "used": sum(g.get("used", 0) for g in got),
                      "agree": round(float(np.median([g.get("agree", 0) for g in got])), 2),
                      "paths": 2 if two else max(g.get("paths", 1) for g in got),
                      "peak": round(float(np.median([g.get("peak", 0) for g in got])), 3)})
    return stations, pairs


CLOSURE_TOL_S = 500e-6       # a pair this far from what the other pairs imply: a wrong correlation peak
RUN_HOP_S = 300e-6           # runs that disagree by more than this on a pair heard different sky-wave paths
FIT_BAD = 4.0                # h at the best point above this (4+ receivers): they disagree (~95 %, 1 degree of freedom)
FIT_GAIN = 4.0               # leaving one receiver out must fit this many times better to be believed


def _lag(pairs, i, j):
    """Measured arrival at i minus arrival at j, or None."""
    for p in pairs:
        if (p["i"], p["j"]) == (i, j):
            return p["lag_s"]
        if (p["i"], p["j"]) == (j, i):
            return -p["lag_s"]
    return None


def drop_inconsistent_pairs(pairs, n):
    """Pairs whose lag contradicts the other pairs: t_i - t_j must equal
    (t_i - t_k) - (t_j - t_k) for every third receiver k. A wrong correlation
    peak breaks that; a receiver's own clock offset does not (it cancels), so this
    finds bad pairs only. Worst triangle first, its weakest pair (lowest correlation
    peak) goes, until every triangle closes: two weak wrong pairs cannot outvote a
    strong right one. -> (kept, dropped)"""
    import itertools
    kept, dropped = list(pairs), []

    def find(i, j):
        return next((p for p in kept if {p["i"], p["j"]} == {i, j}), None)
    while True:
        worst = None
        for i, j, k in itertools.combinations(range(n), 3):
            a, b, c = _lag(kept, i, j), _lag(kept, j, k), _lag(kept, i, k)
            if a is None or b is None or c is None:
                continue
            err = abs(a + b - c)                         # (t_i - t_j) + (t_j - t_k) must be t_i - t_k
            if err > CLOSURE_TOL_S and (worst is None or err > worst[0]):
                worst = (err, [find(i, j), find(j, k), find(i, k)])
        if worst is None:
            return kept, dropped
        bad = min(worst[1], key=lambda p: p.get("peak", 0))
        kept.remove(bad)
        dropped.append(bad)


def combine(stations, pairs):
    """Measured pairs -> the place, robustly. stations: [{"id", "lat", "lon"}];
    pairs: [{"i", "j", "a", "b", "lag_s", "sigma_s", ...}].
    1. Pairs that contradict the others are dropped (wrong correlation peak).
    2. With 4 receivers a disagreement (fit > FIT_BAD) is seen but not placed: any 3
       fit perfectly. The answer is flagged rough.
    3. With 5+, when they disagree, each receiver is left out in turn; if one subset
       fits FIT_GAIN times better, that receiver is named and left out (its clock,
       or its signal path, is off).
    With 3 there is no cross-check at all: the fit is always perfect.
    -> {"lat", "lon", "radius_km", "area", "fit", "pairs", "stations", "excluded",
        "dropped_pairs", "cross_checked", "warning"}"""
    pairs, dropped = drop_inconsistent_pairs(pairs, len(stations))

    def usable(ps, out=()):
        used = {p["i"] for p in ps} | {p["j"] for p in ps}
        return len(used - set(out)) >= 3 and len(ps) >= 2

    def run(ps):
        return locate(stations, [{**p, "sigma_s": math.hypot(p["sigma_s"], SKYWAVE_SIGMA_S)} for p in ps])

    if not usable(pairs):
        raise ValueError(f"only {len(pairs)} receiver pair(s) heard the same signal: TDoA needs 3 receivers "
                         "that all hear it (is the frequency on air, and strong enough at every receiver?)")
    res, excluded = run(pairs), []
    used = sorted({p["i"] for p in pairs} | {p["j"] for p in pairs})
    if len(used) >= 5 and res["fit"] > FIT_BAD:
        tries = []
        for k in used:
            ps = [p for p in pairs if k not in (p["i"], p["j"])]
            left = {p["i"] for p in ps} | {p["j"] for p in ps}
            if usable(ps) and len(left) >= 4:            # 3 left always fit perfectly: not a fair comparison
                tries.append((run(ps), k, ps))
        if tries:
            best, k, ps = min(tries, key=lambda t: t[0]["fit"])
            if best["fit"] * FIT_GAIN < res["fit"]:
                # how far off this receiver's timing is, from the others' place
                off = []
                for p in pairs:
                    if k in (p["i"], p["j"]):
                        r = p["lag_s"] - (_t(stations, p["i"], best) - _t(stations, p["j"], best))
                        off.append(r if p["i"] == k else -r)
                excluded = [{"id": stations[k]["id"], "offset_us": round(float(np.median(off)) * 1e6),
                             "why": "its timing disagrees with the other receivers"}]
                res, pairs = best, ps
    res["pairs"] = [{k: p[k] for k in ("a", "b", "lag_us", "sigma_us", "blocks", "used", "agree", "paths", "runs", "peak")
                     if k in p}
                    for p in pairs]
    res["dropped_pairs"] = [{"a": p["a"], "b": p["b"], "lag_us": p.get("lag_us")} for p in dropped]
    res["excluded"] = excluded
    used = sorted({p["i"] for p in pairs} | {p["j"] for p in pairs})
    res["stations"] = [{"id": stations[i]["id"], "lat": stations[i]["lat"], "lon": stations[i]["lon"]} for i in used]
    res["cross_checked"] = len(used) >= 4
    res["warning"] = (None if not res["cross_checked"] or res["fit"] <= FIT_BAD else
                      f"the {len(used)} receivers disagree (fit {res['fit']:.1f}): one receiver's timing or signal path is "
                      "off" + (" and 4 cannot tell which (5 can): treat this place as rough" if len(used) == 4 else ""))
    if len(used) == 3:
        res["warning"] = "3 receivers: no cross-check (any timing error moves the point unseen)"
    res["alternative"] = _other_hop(stations, pairs, res, run)
    return res


HOP_KM = 2500        # a receiver further than this from the place may hear it via one or two ionospheric hops
LAYER_KM = 300       # virtual height of the reflecting layer (F2)


def _skywave_km(d_km, hops):
    """Path length of a sky wave over d_km of ground in `hops` equal hops off a layer at LAYER_KM."""
    half = d_km / hops / EARTH_KM / 2
    r1 = EARTH_KM + LAYER_KM
    return hops * 2 * math.sqrt(EARTH_KM ** 2 + r1 ** 2 - 2 * EARTH_KM * r1 * math.cos(half))


def _other_hop(stations, pairs, res, run):
    """A far receiver that heard only ONE path may have heard the two-hop one (on
    14 Nov Taiwan, 3200 km from Kajang, switched between one and two hops; its
    two-hop lag put the place 430 km off). When it saw both paths, pair_lag already
    kept the direct one. Here: the place if that receiver had heard it one hop
    earlier, or None when there is no such receiver or that place fits clearly worse.
    -> {"lat", "lon", "receiver", "why"} or None"""
    far = []
    for k, st in enumerate(stations):
        mine = [p for p in pairs if k in (p["i"], p["j"])]
        d = float(great_circle_km(st["lat"], st["lon"], res["lat"], res["lon"]))
        if mine and d > HOP_KM and all(p.get("paths", 1) == 1 and "paths" in p for p in mine):
            far.append((d, k))
    if not far:
        return None
    d, k = max(far)
    alt = res
    for _ in range(4):                     # the extra delay depends on the distance, which depends on the place
        d = float(great_circle_km(stations[k]["lat"], stations[k]["lon"], alt["lat"], alt["lon"]))
        shift = (_skywave_km(d, 2) - _skywave_km(d, 1)) / C_KM_S    # how much later two hops arrive
        alt_pairs = [{**p, "lag_s": p["lag_s"] - shift} if p["i"] == k else
                     {**p, "lag_s": p["lag_s"] + shift} if p["j"] == k else p for p in pairs]
        alt = run(alt_pairs)
    apart = float(great_circle_km(alt["lat"], alt["lon"], res["lat"], res["lon"]))
    if apart < res["radius_km"] or alt["fit"] > max(FIT_BAD, res["fit"] * FIT_GAIN):
        return None
    return {"lat": alt["lat"], "lon": alt["lon"], "receiver": stations[k]["id"], "apart_km": round(apart),
            "why": f"if {stations[k]['id']} ({d:.0f} km away) heard it via two ionospheric hops "
                   f"(~{shift * 1e6:.0f} µs later) rather than one"}


def _t(stations, i, place):
    """Ground-distance travel time from place (a locate() result) to station i, in seconds."""
    s = stations[i]
    return float(great_circle_km(s["lat"], s["lon"], place["lat"], place["lon"])) / C_KM_S
