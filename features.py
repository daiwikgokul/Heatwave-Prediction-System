"""Feature engineering (identical code path for training, batch back-testing and live inference)."""
import numpy as np
import pandas as pd

from .config import CITIES, CITY, CITY_NAMES, DEP_HOT, DEP_HW
from .datagen import distance_matrix
from .imd import base_threshold, classify, doy365, heat_index_c, normal_series

NC = len(CITY_NAMES)
_D = distance_matrix()
_W = 1.0 / np.maximum(_D, 1.0) ** 2
np.fill_diagonal(_W, 0.0)
_W = _W / _W.sum(axis=1, keepdims=True)           # row-normalised inverse-distance^2 weights


def _streak(b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(b), dtype=float)
    run = 0
    for i, v in enumerate(b):
        run = run + 1 if v else 0
        out[i] = min(run, 15)
    return out


def build_feature_frame(df: pd.DataFrame, normals: dict) -> pd.DataFrame:
    """One row per (city, date) with predictors known at the end of that day.

    df: cleaned daily frame (date, city, tmax, tmin, rh, wind, z500) with a complete calendar per city.
    """
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    ff = df[["date", "city", "tmax"]].copy()
    ff["normal"] = normal_series(df, normals)
    dep = df["tmax"].values - ff["normal"].values
    ff["dep"] = dep
    g_city = df["city"].values

    d = df.assign(dep=dep)
    gb = d.groupby("city", sort=False)
    feats = {}
    feats["tmax_l0"] = d["tmax"].values
    feats["tmax_l1"] = gb["tmax"].shift(1).values
    for k in range(0, 7):
        feats[f"dep_l{k}"] = gb["dep"].shift(k).values
    for w in (3, 7, 14, 30):
        feats[f"dep_m{w}"] = gb["dep"].transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).mean()).values
    feats["tmax_max7"] = gb["tmax"].transform(lambda s: s.rolling(7, min_periods=3).max()).values
    feats["dep_std7"] = gb["dep"].transform(lambda s: s.rolling(7, min_periods=3).std()).values
    feats["dep_trend3"] = feats["dep_l0"] - feats["dep_l2"]
    feats["tmin_l0"] = d["tmin"].values
    feats["dtr_l0"] = d["tmax"].values - d["tmin"].values
    feats["rh_l0"] = d["rh"].values
    feats["rh_l1"] = gb["rh"].shift(1).values
    feats["rh_m3"] = gb["rh"].transform(lambda s: s.rolling(3, min_periods=2).mean()).values
    feats["wind_l0"] = d["wind"].values
    feats["wind_m3"] = gb["wind"].transform(lambda s: s.rolling(3, min_periods=2).mean()).values
    feats["hi_l0"] = heat_index_c(d["tmax"].values, d["rh"].values)
    feats["z500_l0"] = d["z500"].values
    feats["z500_chg2"] = d["z500"].values - gb["z500"].shift(2).values
    feats["z500_chg5"] = d["z500"].values - gb["z500"].shift(5).values
    feats["z500_m3"] = gb["z500"].transform(lambda s: s.rolling(3, min_periods=2).mean()).values

    coastal = np.array([CITY[c]["terrain"] == "coastal" for c in g_city])
    cls_today = classify(d["tmax"].values, ff["normal"].values, coastal)
    hot_streak = np.zeros(len(d))
    hw_streak = np.zeros(len(d))
    for city, idx in gb.indices.items():
        hot_streak[idx] = _streak(dep[idx] >= DEP_HOT)
        hw_streak[idx] = _streak(cls_today[idx] >= 2)
    feats["hot_streak"] = hot_streak
    feats["hw_streak"] = hw_streak
    feats["cls_today"] = cls_today.astype(float)

    # calendar / static
    doy = doy365(d["date"])
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    feats["year_f"] = d["date"].dt.year.values + doy / 365.0
    for key in ("lat", "lon", "elev"):
        feats[key] = np.array([CITY[c][key] for c in g_city], dtype=float)
    feats["coastal"] = coastal.astype(float)

    # spatial context: national and neighbour anomalies on the same date
    wide = d.pivot(index="date", columns="city", values="dep")[CITY_NAMES]
    wv = wide.values
    nat_mean = np.nanmean(wv, axis=1)
    nat_max = np.nanmax(wv, axis=1)
    nat_cnt = np.nansum(wv >= DEP_HW, axis=1).astype(float)
    nbr = np.nan_to_num(wv) @ _W.T
    sp = pd.DataFrame({"nat_mean_dep": nat_mean, "nat_max_dep": nat_max, "nat_hw_cnt": nat_cnt}, index=wide.index)
    sp["nat_mean_chg2"] = sp["nat_mean_dep"] - sp["nat_mean_dep"].shift(2)
    sp_ff = sp.reindex(d["date"].values)
    for c in sp_ff.columns:
        feats[c] = sp_ff[c].values
    nbr_df = pd.DataFrame(nbr, index=wide.index, columns=CITY_NAMES)
    nbr_l2 = nbr_df.shift(2)
    ci = np.array([CITY_NAMES.index(c) for c in g_city])
    di = wide.index.get_indexer(d["date"].values)
    feats["nbr_dep"] = nbr_df.values[di, ci]
    feats["nbr_dep_l2"] = nbr_l2.values[di, ci]
    feats["rel_dep"] = feats["dep_l0"] - feats["nbr_dep"]
    zw = d.pivot(index="date", columns="city", values="z500")[CITY_NAMES]
    zv = zw.values
    feats["nat_max_z500"] = pd.Series(np.nanmax(zv, axis=1), index=zw.index).reindex(d["date"].values).values
    znbr = pd.DataFrame(np.nan_to_num(zv) @ _W.T, index=zw.index, columns=CITY_NAMES)
    feats["nbr_z500"] = znbr.values[zw.index.get_indexer(d["date"].values), ci]
    feats["nbr_z500_chg3"] = feats["nbr_z500"] - znbr.shift(3).values[zw.index.get_indexer(d["date"].values), ci]

    out = pd.concat([ff, pd.DataFrame(feats)], axis=1)
    return out


BASE_FEATURES = [
    "tmax_l0", "tmax_l1", "dep_l0", "dep_l1", "dep_l2", "dep_l3", "dep_l4", "dep_l5", "dep_l6",
    "dep_m3", "dep_m7", "dep_m14", "dep_m30", "tmax_max7", "dep_std7", "dep_trend3",
    "tmin_l0", "dtr_l0", "rh_l0", "rh_l1", "rh_m3", "wind_l0", "wind_m3", "hi_l0",
    "z500_l0", "z500_chg2", "z500_chg5", "z500_m3", "nat_max_z500", "nbr_z500", "nbr_z500_chg3",
    "hot_streak", "hw_streak", "cls_today", "doy_sin", "doy_cos", "year_f", "lat", "lon", "elev",
    "coastal", "nat_mean_dep", "nat_max_dep", "nat_hw_cnt", "nat_mean_chg2", "nbr_dep", "nbr_dep_l2",
    "rel_dep",
]
TARGET_FEATURES = ["normal_tgt", "gap_base", "tdoy_sin", "tdoy_cos"]
REG_FEATURES = BASE_FEATURES + TARGET_FEATURES
CLF_FEATURES = BASE_FEATURES + TARGET_FEATURES + ["h"]


def horizon_frame(ff: pd.DataFrame, h: int, normals: dict, with_target: bool = True) -> pd.DataFrame:
    """Attach horizon-specific columns: normal on the target day, target Tmax and class."""
    x = ff.copy()
    tdate = x["date"] + pd.Timedelta(days=h)
    tdoy = doy365(tdate)
    x["target_date"] = tdate
    x["normal_tgt"] = np.nan
    for city, idx in x.groupby("city", sort=False).indices.items():
        x.loc[x.index[idx], "normal_tgt"] = normals[city][tdoy[idx] - 1]
    coastal = x["coastal"].values.astype(bool)
    x["gap_base"] = base_threshold(coastal) - x["normal_tgt"]
    x["tdoy_sin"] = np.sin(2 * np.pi * tdoy / 365.0)
    x["tdoy_cos"] = np.cos(2 * np.pi * tdoy / 365.0)
    x["h"] = h
    if with_target:
        x["tmax_tgt"] = x.groupby("city", sort=False)["tmax"].shift(-h)
        x["dep_tgt"] = x["tmax_tgt"] - x["normal_tgt"]
        x["cls_tgt"] = classify(x["tmax_tgt"].values, x["normal_tgt"].values, coastal)
    return x
