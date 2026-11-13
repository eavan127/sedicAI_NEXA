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
SKYWAVE_SIGMA_S = 100e-6     # ionospheric paths differ from the ground distance: ~30 km of
                             # path length the measurement cannot see, added to every pair


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

def pair_lag(a, b, max_dt_s=0.02, block_s=1.0, min_peak=0.1):
    """Arrival-time difference of one signal at receivers a and b (each (t, z, fs)
    from read_kiwi_wav): how much LATER it reached a than b, in seconds.

    -> {"lag_s", "sigma_s", "blocks", "used", "peak"} or None when no block
    correlates. As hcab14's tdoa_compute_lags.m: 1 s blocks, |cross-correlation|
    peak within +-max_dt_s, parabolic peak fit, outliers dropped, mean and
    spread weighted by peak height squared."""
    (t1, z1, fs1), (t2, z2, fs2) = a, b
    if abs(fs1 - fs2) > 0.01 * fs1:
        raise ValueError(f"sample rates differ ({fs1:.0f} vs {fs2:.0f} S/s)")
    fs = (fs1 + fs2) / 2
    B, M = int(block_s * fs), int(math.ceil(max_dt_s * fs)) + 2
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
        c = np.abs(np.correlate(x1, x2, "valid"))                 # c[m] = |sum x1[m+k] conj(x2[k])|
        norm = math.sqrt(float(np.vdot(x1[M:M + B], x1[M:M + B]).real * np.vdot(x2, x2).real)) or 1.0
        c = c / norm
        base = t1[i] - t2[j]                                       # lag of index m is base + (m - M)/fs
        lag_t = base + (np.arange(len(c)) - M) / fs
        ok = np.flatnonzero(np.abs(lag_t) <= max_dt_s)
        if len(ok) < 3:
            continue
        m = ok[np.argmax(c[ok])]
        if m in (0, len(c) - 1) or c[m] < min_peak:
            continue
        y0, y1, y2 = c[m - 1], c[m], c[m + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        lags.append(base + (m - M + frac) / fs)
        peaks.append(y1)
    if not lags:
        return None
    lags, peaks = np.array(lags), np.array(peaks)
    med = np.median(lags)
    mad = 1.4826 * np.median(np.abs(lags - med))
    keep = np.abs(lags - med) <= max(3 * mad, 2.0 / fs)
    w = peaks[keep] ** 2
    lag = float(np.sum(w * lags[keep]) / np.sum(w))
    var = float(np.sum(w * (lags[keep] - lag) ** 2) / np.sum(w))
    sigma = math.sqrt(max(var, (0.25 / fs) ** 2))                  # never sharper than a quarter sample
    return {"lag_s": lag, "sigma_s": sigma, "blocks": blocks, "used": int(keep.sum()),
            "peak": round(float(np.median(peaks[keep])), 3)}


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
    """recordings: [{"id", "lat", "lon", "t", "z", "fs"}] (3 or 4 receivers).
    -> {"lat", "lon", "radius_km", "area", "fit", "pairs": [...], "stations": [...]}.
    Raises ValueError when fewer than 3 receivers gave a usable correlation."""
    pairs = []
    for i in range(len(recordings)):
        for j in range(i + 1, len(recordings)):
            a, b = recordings[i], recordings[j]
            base_km = float(great_circle_km(a["lat"], a["lon"], b["lat"], b["lon"]))
            r = pair_lag((a["t"], a["z"], a["fs"]), (b["t"], b["z"], b["fs"]),
                         max_dt_s=base_km / C_KM_S + max_dt_extra_s)
            if r:
                pairs.append({"i": i, "j": j, "a": a["id"], "b": b["id"], **r,
                              "lag_us": round(r["lag_s"] * 1e6, 1), "sigma_us": round(r["sigma_s"] * 1e6, 1)})
    used = sorted({p["i"] for p in pairs} | {p["j"] for p in pairs})
    if len(used) < 3 or len(pairs) < 2:
        raise ValueError(f"only {len(pairs)} receiver pair(s) heard the same signal: TDoA needs 3 receivers "
                         "that all hear it (is the frequency on air, and strong enough at every receiver?)")
    stations = [{"lat": r["lat"], "lon": r["lon"]} for r in recordings]
    res = locate(stations, [{**p, "sigma_s": math.hypot(p["sigma_s"], SKYWAVE_SIGMA_S)} for p in pairs])
    res["pairs"] = [{k: p[k] for k in ("a", "b", "lag_us", "sigma_us", "blocks", "used", "peak")} for p in pairs]
    res["stations"] = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in recordings]
    return res
