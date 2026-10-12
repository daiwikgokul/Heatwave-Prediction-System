"""Cached data access + derived tables used by the Streamlit pages."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from .config import ALERT_LEVELS, CITY, DATA, MONTH_TO_SEASON, SEVERITY

P = DATA / "processed"


def ready() -> bool:
    need = ["daily_enriched.parquet", "predictions.parquet", "metrics.json", "aws_hourly.parquet"]
    return all((P / n).exists() for n in need) and (DATA.parent / "models" / "heatwave_models.joblib").exists()


@st.cache_data(show_spinner=False)
def daily() -> pd.DataFrame:
    df = pd.read_parquet(P / "daily_enriched.parquet")
    df["region"] = df["city"].map(lambda c: CITY[c]["region"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(MONTH_TO_SEASON)
    return df


@st.cache_data(show_spinner=False)
def raw_daily() -> pd.DataFrame:
    return pd.read_csv(DATA / "raw" / "synthetic_imd_daily_raw.csv", parse_dates=["date"])


@st.cache_data(show_spinner=False)
def predictions() -> pd.DataFrame:
    return pd.read_parquet(P / "predictions.parquet")


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return json.load(open(P / "metrics.json"))


@st.cache_data(show_spinner=False)
def quality() -> dict:
    return json.load(open(P / "quality_report.json"))


@st.cache_data(show_spinner=False)
def stations() -> pd.DataFrame:
    return pd.read_csv(P / "aws_stations.csv")


@st.cache_data(show_spinner=False)
def aws() -> pd.DataFrame:
    df = pd.read_parquet(P / "aws_hourly.parquet")
    df["date"] = df["ts"].dt.floor("D")
    return df


@st.cache_resource(show_spinner="Loading models ...")
def engine():
    from .models import Engine
    return Engine()


# --------------------------------------------------------------------------- derived views
def forecast_at(asof) -> pd.DataFrame:
    """Pre-computed (back-tested) forecast issued at the end of `asof` for all cities and leads."""
    p = predictions()
    f = p[p["issue_date"] == pd.Timestamp(asof)].copy()
    f["dep_pred"] = f["tmax_pred"] - f["normal_tgt"]
    return f.sort_values(["city", "h"]).reset_index(drop=True)


def city_status(fc: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    """One row per city summarising the next `lookahead` days."""
    rows = []
    for city, g in fc[fc["h"] <= lookahead].groupby("city"):
        worst = int(g["alert"].max())
        first = int(g.loc[g["alert"] >= max(worst, 1), "h"].min()) if worst >= 1 else 0
        pk = g.loc[g["hri"].idxmax()]
        tmx = g.loc[g["tmax_pred"].idxmax()]
        rows.append(dict(
            city=city, region=CITY[city]["region"], lat=CITY[city]["lat"], lon=CITY[city]["lon"],
            pop=CITY[city]["pop"], alert=worst, first_day=first, peak_hri=float(pk["hri"]), peak_hri_h=int(pk["h"]),
            peak_tmax=float(tmx["tmax_pred"]), peak_tmax_h=int(tmx["h"]),
            peak_dep=float(tmx["tmax_pred"] - tmx["normal_tgt"]),
            p_hw_max=float((g["p2"] + g["p3"]).max()), p_sev_max=float(g["p3"].max()),
            n_alert_days=int((g["alert"] >= 2).sum()),
            colour=ALERT_LEVELS[worst]["hex"], label=ALERT_LEVELS[worst]["colour"],
            name=ALERT_LEVELS[worst]["name"],
        ))
    out = pd.DataFrame(rows)
    return out.sort_values(["alert", "peak_hri"], ascending=False).reset_index(drop=True)


def observed_on(asof, city=None) -> pd.DataFrame:
    d = daily()
    x = d[d["date"] == pd.Timestamp(asof)]
    return x if city is None else x[x["city"] == city]


def heatwave_spells(df: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    """Spells of >= min_len consecutive heatwave days (class >= 2) per city."""
    rows = []
    for city, g in df.sort_values("date").groupby("city"):
        flag = (g["cls"].values >= 2)
        dates = g["date"].values
        tm = g["tmax"].values
        i = 0
        while i < len(flag):
            if flag[i]:
                j = i
                while j + 1 < len(flag) and flag[j + 1]:
                    j += 1
                if j - i + 1 >= min_len:
                    rows.append(dict(city=city, start=pd.Timestamp(dates[i]), end=pd.Timestamp(dates[j]),
                                     days=j - i + 1, peak_tmax=float(tm[i:j + 1].max()),
                                     severe_days=int((g["cls"].values[i:j + 1] >= 3).sum())))
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def severity_name(c: int) -> str:
    return SEVERITY[int(c)]
