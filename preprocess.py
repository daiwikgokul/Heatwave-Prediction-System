"""Phase I - data quality assessment, cleaning and transformation."""
import numpy as np
import pandas as pd

from .imd import doy365

GRD_MISSING = 99.9


def clean_daily(raw: pd.DataFrame):
    """Clean the raw city-day archive. Returns (clean_df, qa_report dict)."""
    qa = {"rows_raw": int(len(raw))}
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["city", "date"], keep="first")
    qa["duplicates_removed"] = int(before - len(df))

    # 2. reindex to the full calendar -> reveals missing days
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    parts, missing_days = [], 0
    for city, g in df.groupby("city"):
        g = g.set_index("date").reindex(full)
        missing_days += int(g["city"].isna().sum())   # calendar days absent from the archive
        g["city"] = city
        g.index.name = "date"
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)
    qa["missing_days_added"] = missing_days

    # 3. sentinel codes and physical-range checks
    qa["sentinel_99_9"] = int((df["tmax"] >= 60).sum() + (df["tmin"] >= 60).sum())
    df.loc[df["tmax"] >= 60, "tmax"] = np.nan
    df.loc[df["tmin"] >= 60, "tmin"] = np.nan
    bad_t = (df["tmax"] < 0) | (df["tmax"] > 52)
    bad_rh = (df["rh"] < 1) | (df["rh"] > 100)
    bad_w = df["wind"] < 0
    bad_z = df["z500"].abs() > 300
    qa["range_violations"] = int(bad_t.sum() + bad_rh.sum() + bad_w.sum() + bad_z.sum())
    df.loc[bad_z, "z500"] = np.nan
    df.loc[bad_t, "tmax"] = np.nan
    df.loc[bad_rh, "rh"] = np.nan
    df.loc[bad_w, "wind"] = np.nan

    # 4. logical check: Tmin must not exceed Tmax (swap errors)
    swap = df["tmin"] > df["tmax"]
    qa["tmin_gt_tmax"] = int(swap.sum())
    df.loc[swap, ["tmax", "tmin"]] = np.nan

    # 5. isolated spike detection on Tmax (deviation from mean of neighbours)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    g = df.groupby("city")["tmax"]
    nb = 0.5 * (g.shift(1) + g.shift(-1))
    spike = (df["tmax"] - nb).abs() > 7.5
    qa["spikes_flagged"] = int(spike.sum())
    df.loc[spike, "tmax"] = np.nan

    # 6. imputation: time interpolation for short gaps, climatology+persistence for long gaps
    qa["missing_before_impute"] = {c: int(df[c].isna().sum()) for c in ["tmax", "tmin", "rh", "wind", "z500"]}
    d = doy365(df["date"])
    for col in ["tmax", "tmin", "rh", "wind", "z500"]:
        out = df[col].copy()
        for city, idx in df.groupby("city").indices.items():
            s = pd.Series(df.loc[idx, col].values, index=df.loc[idx, "date"])
            s_i = s.interpolate(method="time", limit=5, limit_direction="both")
            if s_i.isna().any():  # long gaps -> day-of-year mean
                clim = pd.Series(s.values).groupby(d[idx]).transform("mean").values
                s_i = s_i.fillna(pd.Series(clim, index=s.index))
            out.iloc[idx] = s_i.values
        df[col] = out.round(1)
    qa["rows_clean"] = int(len(df))
    qa["missing_after_impute"] = {c: int(df[c].isna().sum()) for c in ["tmax", "tmin", "rh", "wind", "z500"]}
    n_cells = len(df) * 5
    qa["pct_cells_repaired"] = round(100 * sum(qa["missing_before_impute"].values()) / n_cells, 3)
    return df[["date", "city", "tmax", "tmin", "rh", "wind", "z500"]], qa


def qc_aws(aws: pd.DataFrame) -> pd.DataFrame:
    """Quality-control flags for hourly AWS data: 0 ok, 1 range, 2 spike, 3 missing."""
    aws = aws.sort_values(["station_id", "ts"]).reset_index(drop=True)
    flag = np.zeros(len(aws), dtype="int8")
    t = aws["temp_c"]
    miss = t.isna() | aws["rh_pct"].isna()
    rng_bad = (t < 0) | (t > 55) | (aws["rh_pct"] < 0) | (aws["rh_pct"] > 100)
    g = aws.groupby("station_id")["temp_c"]
    nb = 0.5 * (g.shift(1) + g.shift(-1))
    spike = (t - nb).abs() > 5.0
    flag[spike.values] = 2
    flag[rng_bad.values] = 1
    flag[miss.values] = 3
    aws["qc_flag"] = flag
    return aws
