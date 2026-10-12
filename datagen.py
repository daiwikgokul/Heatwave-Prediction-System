"""Synthetic meteorological database for 10 Indian cities + simulated IoT AWS feed.

Everything here is SYNTHETIC. Seasonal cycles are fitted through approximate monthly
means; heatwaves are injected as spatially coherent synoptic events (Gaussian footprint
around a random centre) so that neighbouring cities heat up together, plus local events.
A warming trend and growing event frequency mimic climate change.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .config import (AWS_SITES, AWS_START, CITIES, DATA_END, DATA_START, DTR_MONTHLY, SIGMA_MONTHLY,
                     STATION_TYPES)
from .imd import doy365, heat_index_c

MID = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5])
DAYS = np.arange(1, 366)


def _periodic_curve(monthly) -> np.ndarray:
    """Smooth periodic daily curve (365 values) through 12 monthly means."""
    x = np.concatenate([MID - 365, MID, MID + 365])
    y = np.tile(np.asarray(monthly, dtype=float), 3)
    return CubicSpline(x, y)(DAYS)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def distance_matrix() -> np.ndarray:
    lat = np.array([c["lat"] for c in CITIES])
    lon = np.array([c["lon"] for c in CITIES])
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def _bell(length: int) -> np.ndarray:
    t = (np.arange(length) + 0.5) / length
    return np.sin(np.pi * t) ** 1.4


def generate_clean(seed: int = 2026) -> pd.DataFrame:
    """Return the 'ground truth' daily database (one row per city-day)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(DATA_START, DATA_END, freq="D")
    n, nc = len(dates), len(CITIES)
    d365 = doy365(dates)
    years = dates.year.values
    year0 = years.min()
    lat = np.array([c["lat"] for c in CITIES])
    lon = np.array([c["lon"] for c in CITIES])
    D = distance_matrix()

    # ---- background weather noise: AR(1) with spatially correlated innovations
    L = np.linalg.cholesky(np.exp(-D / 700.0) + 1e-6 * np.eye(nc))
    z = rng.standard_normal((n, nc)) @ L.T
    phi = 0.72
    e = np.zeros((n, nc))
    for t in range(1, n):
        e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * z[t]

    sig_curve = _periodic_curve(SIGMA_MONTHLY)
    clim_t, clim_rh, clim_dtr = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros((n, nc))
    sigma = np.zeros((n, nc))
    for j, c in enumerate(CITIES):
        clim_t[:, j] = _periodic_curve(c["tmax"])[d365 - 1]
        clim_rh[:, j] = _periodic_curve(c["rh"])[d365 - 1]
        clim_dtr[:, j] = _periodic_curve(DTR_MONTHLY)[d365 - 1] * c["dtr"]
        sigma[:, j] = sig_curve[d365 - 1] * c["sig"]
    anomaly = sigma * e

    # ---- synoptic precursor: 500-hPa height anomaly proxy ("ridge index", metres)
    zr = rng.standard_normal((n, nc)) @ L.T
    rn = np.zeros((n, nc))
    for t in range(1, n):
        rn[t] = 0.85 * rn[t - 1] + np.sqrt(1 - 0.85 ** 2) * zr[t]
    ridge = 13.0 * rn
    ridge_ev = np.zeros((n, nc))

    # ---- climate signal: warming trend + inter-annual variability
    yr_common = rng.normal(0, 0.45, size=years.max() - year0 + 1)
    yr_city = rng.normal(0, 0.25, size=(years.max() - year0 + 1, nc))
    trend = 0.035 * (years - 2015.0)
    climate = trend[:, None] + yr_common[years - year0][:, None] + yr_city[years - year0]

    # ---- heatwave events -----------------------------------------------------
    events = np.zeros((n, nc))
    start_idx = {y: np.where(years == y)[0][0] for y in np.unique(years)}

    def add_event(day_idx, dur, amp, weights):
        sl = slice(day_idx, min(day_idx + dur, n))
        shape = _bell(dur)[: sl.stop - sl.start]
        events[sl] += shape[:, None] * (amp * weights)[None, :]

    for y in np.unique(years):
        ndays_y = int((years == y).sum())
        # synoptic (regional) events
        for _ in range(rng.poisson(4.4 + 0.07 * (y - year0))):
            doy = 68 + 112 * rng.beta(2.6, 2.2)
            dur = int(np.clip(rng.gamma(4.0, 1.3) + 2, 3, 12))
            amp = 3.0 + rng.gamma(2.0, 0.75) + 0.03 * (y - year0)
            clat, clon, R = rng.uniform(17, 29), rng.uniform(71, 86), rng.uniform(450, 1000)
            dist = haversine_km(lat, lon, clat, clon)
            foot = np.exp(-(dist / R) ** 2)
            w = foot * np.array([c["suscept"] for c in CITIES])
            idx = start_idx[y] + int(doy)
            if idx < n:
                # the ridge builds `lead` days before the heat arrives; ~25% of ridges under-deliver
                lead = int(rng.integers(3, 9))
                deliver = rng.uniform(0.25, 0.6) if rng.random() < 0.25 else rng.uniform(0.8, 1.25)
                add_event(idx, dur, amp * deliver, w)
                s0 = max(idx - lead, 0)
                ln = min(idx + dur, n) - s0
                ridge_ev[s0: s0 + ln] += _bell(lead + dur)[:ln, None] * (11.0 * amp * rng.uniform(0.85, 1.15) * foot)[None, :]
        # local (city-scale) events
        for j, c in enumerate(CITIES):
            for _ in range(rng.poisson(0.45 * c["suscept"])):
                doy = 70 + 110 * rng.beta(2.4, 2.2)
                dur = int(np.clip(rng.gamma(3.0, 1.2) + 2, 3, 8))
                amp = 2.4 + rng.gamma(2.0, 0.6)
                w = np.zeros(nc)
                w[j] = c["suscept"] if c["suscept"] > 0.5 else 0.6
                idx = start_idx[y] + int(doy)
                if idx < n:
                    add_event(idx, dur, amp, w)
        # winter cold spells (north India)
        for _ in range(rng.poisson(1.5)):
            doy = rng.choice([rng.uniform(1, 50), rng.uniform(340, 364)])
            dur = int(np.clip(rng.gamma(3.0, 1.0) + 2, 3, 7))
            amp = -(2.8 + rng.gamma(2.0, 0.8))
            dist = haversine_km(lat, lon, rng.uniform(22, 30), rng.uniform(73, 85))
            w = np.exp(-(dist / 800.0) ** 2) * np.where(lat > 20, 1.0, 0.25)
            idx = start_idx[y] + int(doy)
            if idx < n:
                add_event(idx, dur, amp, w)

    z500 = ridge + ridge_ev + 3.0 * anomaly + rng.normal(0, 4.0, size=(n, nc))
    tmax = clim_t + climate + anomaly + events
    tmax = np.clip(tmax, clim_t - 12, 48.8)
    dep = tmax - clim_t

    rh = clim_rh - 1.5 * (dep - climate) + rng.normal(0, 5.5, size=(n, nc))
    rh = np.clip(rh, 8, 98)
    dtr = np.clip(clim_dtr + 0.30 * (dep - climate) + rng.normal(0, 0.9, size=(n, nc)), 3.0, None)
    tmin = tmax - dtr
    mon_w = np.exp(-(((d365 - 200) / 42.0) ** 2))
    wind_base = np.array([c["wind"] for c in CITIES])
    wind = wind_base[None, :] * (1 + 0.35 * mon_w[:, None]) * rng.lognormal(0, 0.32, size=(n, nc))

    frames = []
    for j, c in enumerate(CITIES):
        frames.append(pd.DataFrame({
            "date": dates, "city": c["name"], "tmax": tmax[:, j], "tmin": tmin[:, j],
            "rh": rh[:, j], "wind": wind[:, j], "z500": z500[:, j], "clim_true": clim_t[:, j],
        }))
    df = pd.concat(frames, ignore_index=True)
    for col in ["tmax", "tmin", "rh", "wind", "z500"]:
        df[col] = df[col].round(1)
    return df


# --------------------------------------------------------------------------- raw defects
def make_raw(clean: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Inject realistic archive defects: IMD-style 99.9 sentinels, NaNs, spikes,
    duplicated rows, whole missing days and swapped Tmin/Tmax."""
    rng = np.random.default_rng(seed)
    raw = clean.drop(columns=["clim_true"]).copy()
    n = len(raw)

    # scattered missing values
    for col, rate in [("tmax", 0.006), ("tmin", 0.006), ("rh", 0.004), ("wind", 0.004), ("z500", 0.004)]:
        idx = rng.choice(n, int(n * rate), replace=False)
        if col in ("tmax", "tmin"):
            half = len(idx) // 2
            raw.loc[idx[:half], col] = 99.9          # GRD missing-value code
            raw.loc[idx[half:], col] = np.nan
        else:
            raw.loc[idx, col] = np.nan
    # isolated spikes in Tmax
    idx = rng.choice(n, int(n * 0.0015), replace=False)
    raw.loc[idx, "tmax"] = raw.loc[idx, "tmax"] + rng.choice([-1, 1], len(idx)) * rng.uniform(9, 15, len(idx))
    # swapped tmin/tmax
    idx = rng.choice(n, int(n * 0.0006), replace=False)
    raw.loc[idx, ["tmax", "tmin"]] = raw.loc[idx, ["tmin", "tmax"]].values
    # impossible RH / wind
    idx = rng.choice(n, int(n * 0.0005), replace=False)
    raw.loc[idx, "rh"] = rng.choice([-5.0, 112.0, 130.0], len(idx))
    # whole missing days (2 gaps per city)
    drop = []
    for city in raw["city"].unique():
        rows = np.where(raw["city"].values == city)[0]
        for _ in range(2):
            s = rng.integers(200, len(rows) - 50)
            drop.extend(rows[s: s + rng.integers(3, 9)])
    raw = raw.drop(index=drop)
    # duplicated rows
    dup = raw.sample(frac=0.001, random_state=seed)
    raw = pd.concat([raw, dup], ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)
    return raw


# --------------------------------------------------------------------------- AWS / IoT
def station_registry() -> pd.DataFrame:
    rows = []
    for c in CITIES:
        for k, (name, stype, la, lo) in enumerate(AWS_SITES[c["name"]], start=1):
            rows.append(dict(station_id=f"{c['code']}-{k:02d}", city=c["name"], name=name, type=stype,
                             lat=la, lon=lo, elevation_m=c["elev"], installed=AWS_START,
                             sensor="Digital T/RH sensor + cup anemometer", power="Solar + Li-ion",
                             comms="4G / LoRaWAN gateway"))
    return pd.DataFrame(rows)


def _diurnal_f() -> np.ndarray:
    """Normalised diurnal curve f(h): 0 at 06:00 (Tmin), 1 at 15:00 (Tmax)."""
    h = np.arange(24)
    f = np.zeros(24)
    for i, hh in enumerate(h):
        x = hh if hh >= 6 else hh + 24
        if x <= 15:
            f[i] = 0.5 - 0.5 * np.cos(np.pi * (x - 6) / 9.0)
        else:
            f[i] = 0.5 + 0.5 * np.cos(np.pi * (x - 15) / 15.0)
    return f


def generate_aws(clean: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Hourly IoT readings for every station from AWS_START to DATA_END."""
    rng = np.random.default_rng(seed)
    reg = station_registry()
    f = _diurnal_f()
    sub = clean[clean["date"] >= AWS_START]
    cols = {}
    frames = []
    spike_stations = set(rng.choice(reg["station_id"], 3, replace=False))
    weak_battery = set(rng.choice(reg["station_id"], 2, replace=False))
    for _, st in reg.iterrows():
        g = sub[sub["city"] == st["city"]].sort_values("date")
        nd = len(g)
        t_off, rh_off, _ = STATION_TYPES[st["type"]]
        dep = (g["tmax"] - g["clim_true"]).values
        amp = 1 + 0.10 * np.clip(dep, 0, 8)           # urban heat island intensifies in heatwaves
        tmax_s = g["tmax"].values + t_off * amp + rng.normal(0, 0.45, nd) + rng.normal(0, 0.2)
        tmin_s = g["tmin"].values + 0.5 * t_off + rng.normal(0, 0.4, nd)
        rh_aft = np.clip(g["rh"].values + rh_off + rng.normal(0, 3, nd), 6, 95)
        rh_dawn = np.clip(rh_aft + 30 + rng.normal(0, 4, nd), 20, 98)
        temp = tmin_s[:, None] + (tmax_s - tmin_s)[:, None] * f[None, :] + rng.normal(0, 0.25, (nd, 24))
        rh = rh_aft[:, None] + (rh_dawn - rh_aft)[:, None] * (1 - f[None, :]) + rng.normal(0, 1.8, (nd, 24))
        rh = np.clip(rh, 5, 100)
        wind = g["wind"].values[:, None] * (0.7 + 0.6 * f[None, :]) * rng.lognormal(0, 0.28, (nd, 24))
        hi = heat_index_c(temp, rh)

        solar = np.clip(np.sin(np.pi * (np.arange(24) - 6) / 12.0), 0, None)
        doy = g["date"].dt.dayofyear.values
        cloud = 1 - 0.55 * np.exp(-(((doy - 200) / 45.0) ** 2))
        drift = np.linspace(0, 0.07 if st["station_id"] not in weak_battery else 0.55, nd)
        battery = (3.82 - drift[:, None] + 0.34 * solar[None, :] * cloud[:, None]
                   + rng.normal(0, 0.012, (nd, 24)))
        rssi = (-62 - rng.uniform(0, 24) + rng.normal(0, 2.5, (nd, 24))
                - 5 * (1 - cloud)[:, None])

        ts = (g["date"].values[:, None] + (np.arange(24) * np.timedelta64(1, "h"))[None, :]).ravel()
        df = pd.DataFrame({
            "ts": ts, "station_id": st["station_id"],
            "temp_c": temp.ravel().astype("float32"), "rh_pct": rh.ravel().astype("float32"),
            "wind_kmh": wind.ravel().astype("float32"), "heat_index_c": hi.ravel().astype("float32"),
            "battery_v": battery.ravel().astype("float32"), "rssi_dbm": rssi.ravel().astype("float32"),
        })
        n = len(df)
        # outages: all measurements lost
        nout = rng.poisson(nd / 28.0)
        for _ in range(nout):
            s = rng.integers(0, n - 80)
            ln = int(min(rng.exponential(10) + 2, 72))
            df.loc[s: s + ln, ["temp_c", "rh_pct", "wind_kmh", "heat_index_c", "battery_v", "rssi_dbm"]] = np.nan
        # spike faults on a few stations
        if st["station_id"] in spike_stations:
            idx = rng.choice(n, 14, replace=False)
            df.loc[idx, "temp_c"] = (df.loc[idx, "temp_c"].values + rng.choice([-1, 1], 14) * rng.uniform(9, 14, 14)).astype("float32")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out
