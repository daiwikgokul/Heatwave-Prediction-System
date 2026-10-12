"""IMD heatwave criteria, heat index and climatological normals."""
import numpy as np
import pandas as pd

from .config import (CITIES, CITY, DEP_HOT, DEP_HW, DEP_SEVERE, IMD, MONTH_TO_SEASON, SEVERITY)


# --------------------------------------------------------------------------- calendar
def doy365(dates) -> np.ndarray:
    """Day-of-year on a fixed 365-day calendar (29 Feb is folded onto 28 Feb)."""
    d = pd.DatetimeIndex(dates)
    doy = d.dayofyear.values.copy()
    leap_after_feb = d.is_leap_year & (d.month > 2)
    doy = doy - leap_after_feb.astype(int)
    doy[(d.month == 2) & (d.day == 29)] = 59
    return doy


def season_of(dates) -> pd.Series:
    d = pd.DatetimeIndex(dates)
    return pd.Series([MONTH_TO_SEASON[m] for m in d.month], index=d)


# --------------------------------------------------------------------------- heat index
def heat_index_c(t_c, rh):
    """NWS Rothfusz heat index in deg C (vectorised). Falls back to the simple
    Steadman formula when the regression is outside its validity range."""
    t_c = np.asarray(t_c, dtype=float)
    rh = np.clip(np.asarray(rh, dtype=float), 1, 100)
    T = t_c * 9.0 / 5.0 + 32.0
    simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * T + 10.14333127 * rh - 0.22475541 * T * rh - 0.00683783 * T * T
          - 0.05481717 * rh * rh + 0.00122874 * T * T * rh + 0.00085282 * T * rh * rh
          - 0.00000199 * T * T * rh * rh)
    adj1 = np.where((rh < 13) & (T > 80) & (T < 112),
                    ((13 - rh) / 4.0) * np.sqrt(np.clip((17 - np.abs(T - 95.0)) / 17.0, 0, None)), 0.0)
    adj2 = np.where((rh > 85) & (T > 80) & (T < 87), ((rh - 85) / 10.0) * ((87 - T) / 5.0), 0.0)
    hi = hi - adj1 + adj2
    out = np.where(simple >= 80.0, hi, simple)
    return (out - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------- IMD classification
def classify(tmax, normal, coastal) -> np.ndarray:
    """Daily severity class: 0 Normal, 1 Hot Day, 2 Heatwave, 3 Severe Heatwave.

    Plains: Tmax >= 40 C and departure >= 4.5 C (severe >= 6.5 C), or Tmax >= 45 C (severe >= 47 C)
    Coastal: Tmax >= 37 C and departure >= 4.5 C (severe >= 6.5 C)
    Hot day (our watch tier): same Tmax floor with departure >= 3.0 C.
    """
    tmax = np.asarray(tmax, dtype=float)
    normal = np.asarray(normal, dtype=float)
    coastal = np.asarray(coastal, dtype=bool)
    dep = tmax - normal
    base = np.where(coastal, IMD["coastal"]["base"], IMD["plains"]["base"])
    floor_ok = tmax >= base
    hot = floor_ok & (dep >= DEP_HOT)
    hw = (floor_ok & (dep >= DEP_HW)) | (~coastal & (tmax >= IMD["plains"]["abs_hw"]))
    sev = (floor_ok & (dep >= DEP_SEVERE)) | (~coastal & (tmax >= IMD["plains"]["abs_severe"]))
    cls = np.zeros(tmax.shape, dtype=int)
    cls[hot] = 1
    cls[hw] = 2
    cls[sev] = 3
    cls[np.isnan(tmax) | np.isnan(normal)] = -1
    return cls


def base_threshold(coastal) -> np.ndarray:
    return np.where(np.asarray(coastal, dtype=bool), IMD["coastal"]["base"], IMD["plains"]["base"])


def label(cls: int) -> str:
    return SEVERITY[int(cls)]


# --------------------------------------------------------------------------- normals
def compute_normals(df: pd.DataFrame, years=None, sigma_days: float = 6.0) -> dict:
    """Per-city smoothed day-of-year climatology of Tmax (365 values, index = doy365-1)."""
    out = {}
    for city, g in df.groupby("city"):
        g = g.dropna(subset=["tmax"])
        if years is not None:
            g = g[g["date"].dt.year.isin(years)]
        d = doy365(g["date"])
        s = np.bincount(d - 1, weights=g["tmax"].values, minlength=365)
        c = np.bincount(d - 1, minlength=365)
        raw = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        # circular Gaussian smoothing
        k = np.arange(-30, 31)
        w = np.exp(-0.5 * (k / sigma_days) ** 2)
        w /= w.sum()
        ext = np.concatenate([raw[-30:], raw, raw[:30]])
        sm = np.convolve(ext, w, mode="same")[30:-30]
        out[city] = sm
    return out


def normal_for(city: str, dates, normals: dict) -> np.ndarray:
    return normals[city][doy365(dates) - 1]


def normal_series(df: pd.DataFrame, normals: dict) -> np.ndarray:
    """Normal for every row of a (city, date) frame."""
    out = np.full(len(df), np.nan)
    d = doy365(df["date"])
    for city, idx in df.groupby("city").indices.items():
        out[idx] = normals[city][d[idx] - 1]
    return out


def is_coastal(city: str) -> bool:
    return CITY[city]["terrain"] == "coastal"
