"""Run after build_all.py:  python scripts/evaluate.py
Back-tests the models on the hold-out period (2024-01-01 ->), verifies forecasts against
IMD-style rules and AWS observations, and writes data/processed/{predictions.parquet,metrics.json}."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from hw.config import AWS_START, CITY, DATA, HORIZONS, IMD, MONTH_TO_SEASON, TRAIN_END, VAL_END
from hw.features import REG_FEATURES, build_feature_frame, horizon_frame
from hw.imd import classify
from hw.models import Engine, hri

TEST_START = "2024-01-01"
OUT = DATA / "processed"


def cat_stats(y, a):
    y, a = np.asarray(y, bool), np.asarray(a, bool)
    tp, fp, fn, tn = int((a & y).sum()), int((a & ~y).sum()), int((~a & y).sum()), int((~a & ~y).sum())
    pod = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    csi = tp / max(tp + fp + fn, 1)
    prec = 1 - far
    f1 = 2 * prec * pod / max(prec + pod, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, pod=round(pod, 3), far=round(far, 3), csi=round(csi, 3),
                f1=round(f1, 3), freq_bias=round((tp + fp) / max(tp + fn, 1), 2))


def main():
    clean = pd.read_parquet(OUT / "daily_clean.parquet")
    eng = Engine()
    ff = build_feature_frame(clean, eng.normals)
    full = eng.predict_frame(ff)
    full["issue_date"] = full["date"]
    metrics = {}

    # ---------------------------------------------------------------- save test predictions for the app
    test = full[full["issue_date"] >= TEST_START].copy()
    keep = ["city", "issue_date", "h", "target_date", "tmax_pred", "tmax_lo", "tmax_hi", "normal_tgt",
            "p0", "p1", "p2", "p3", "hri", "alert", "pred_class", "tmax_tgt", "cls_tgt", "pc_ge2", "pg_ge2"]
    test[keep].astype({c: "float32" for c in ["tmax_pred", "tmax_lo", "tmax_hi", "normal_tgt", "p0", "p1", "p2", "p3",
                                              "hri", "tmax_tgt", "pc_ge2", "pg_ge2"]}).to_parquet(
        OUT / "predictions.parquet", index=False)

    # ---------------------------------------------------------------- dataset summary
    clean_ = clean.copy()
    clean_["normal"] = ff.set_index(["city", "date"]).loc[list(zip(clean_["city"], clean_["date"])), "normal"].values
    clean_["coastal"] = clean_["city"].map(lambda c: CITY[c]["terrain"] == "coastal")
    clean_["cls"] = classify(clean_["tmax"], clean_["normal"], clean_["coastal"])
    metrics["dataset"] = dict(
        rows=int(len(clean_)), cities=int(clean_["city"].nunique()),
        start=str(clean_["date"].min().date()), end=str(clean_["date"].max().date()),
        class_counts={int(k): int(v) for k, v in clean_["cls"].value_counts().sort_index().items()},
        train_end=TRAIN_END, val_end=VAL_END, test_start=TEST_START,
        test_days=int(test["issue_date"].nunique()),
        test_hw_city_days=int(((clean_["date"] >= TEST_START) & (clean_["cls"] >= 2)).sum()),
        aws_start=AWS_START)

    ev = test[test["tmax_tgt"].notna()].copy()
    ev["season"] = ev["target_date"].dt.month.map(MONTH_TO_SEASON)
    # baselines (all predictors are already columns of the horizon frame)
    ev["tmax_persist"] = ev["tmax_l0"]                       # tomorrow = today
    ev["tmax_persist_anom"] = ev["normal_tgt"] + ev["dep_l0"]  # today's anomaly persists
    ev["tmax_clim"] = ev["normal_tgt"]                       # climatology

    # ---------------------------------------------------------------- Tmax regression skill
    rows = []
    for scope, sub in [("All seasons", ev), ("Pre-monsoon (Mar-May)", ev[ev["season"] == "Pre-monsoon"])]:
        for h in HORIZONS:
            x = sub[sub["h"] == h]
            r = dict(scope=scope, h=h, n=int(len(x)))
            for name, col in [("model", "tmax_pred"), ("persistence", "tmax_persist"),
                              ("persist_anomaly", "tmax_persist_anom"), ("climatology", "tmax_clim")]:
                err = x[col] - x["tmax_tgt"]
                r[f"{name}_mae"] = round(float(err.abs().mean()), 3)
                r[f"{name}_rmse"] = round(float(np.sqrt((err ** 2).mean())), 3)
                if name == "model":
                    r["bias"] = round(float(err.mean()), 3)
                    ss_res = float((err ** 2).sum())
                    ss_tot = float(((x["tmax_tgt"] - x["tmax_tgt"].mean()) ** 2).sum())
                    r["r2"] = round(1 - ss_res / ss_tot, 4)
            r["coverage90"] = round(float(((x["tmax_tgt"] >= x["tmax_lo"]) & (x["tmax_tgt"] <= x["tmax_hi"])).mean()), 3)
            r["skill_vs_persist_anom"] = round(1 - r["model_mae"] / r["persist_anomaly_mae"], 3)
            rows.append(r)
    metrics["regression"] = rows

    # extreme-day Tmax error (only days that were actually heatwave days)
    hw_days = ev[ev["cls_tgt"] >= 2]
    metrics["regression_hw_days"] = [dict(h=h, n=int((hw_days["h"] == h).sum()),
                                          mae=round(float((hw_days[hw_days["h"] == h]["tmax_pred"] -
                                                           hw_days[hw_days["h"] == h]["tmax_tgt"]).abs().mean()), 3),
                                          bias=round(float((hw_days[hw_days["h"] == h]["tmax_pred"] -
                                                            hw_days[hw_days["h"] == h]["tmax_tgt"]).mean()), 3))
                                     for h in HORIZONS]

    # ---------------------------------------------------------------- heatwave detection skill
    # climatological probability by city x month from training labels
    tr = clean_[clean_["date"] <= TRAIN_END].copy()
    tr["m"] = tr["date"].dt.month
    clim_rate = tr.groupby(["city", "m"])["cls"].apply(lambda s: (s >= 2).mean()).rename("clim_p").reset_index()
    ev["m"] = ev["target_date"].dt.month
    ev = ev.merge(clim_rate, left_on=["city", "m"], right_on=["city", "m"], how="left")

    det = []
    for h in HORIZONS:
        x = ev[(ev["h"] == h) & (ev["cls_tgt"] >= 0)]
        y = (x["cls_tgt"] >= 2).values
        p = (x["p2"] + x["p3"]).values
        row = dict(h=h, n=int(len(x)), positives=int(y.sum()))
        row["auc"] = round(float(roc_auc_score(y, p)), 3)
        row["ap"] = round(float(average_precision_score(y, p)), 3)
        row["auc_classifier"] = round(float(roc_auc_score(y, x["pc_ge2"])), 3)
        row["auc_regression"] = round(float(roc_auc_score(y, x["pg_ge2"])), 3)
        row["ap_classifier"] = round(float(average_precision_score(y, x["pc_ge2"])), 3)
        row["ap_regression"] = round(float(average_precision_score(y, x["pg_ge2"])), 3)
        bs = float(((p - y) ** 2).mean())
        bs_clim = float(((x["clim_p"].values - y) ** 2).mean())
        row["brier"] = round(bs, 4)
        row["brier_skill"] = round(1 - bs / bs_clim, 3)
        row["alert_orange_plus"] = cat_stats(y, x["alert"].values >= 2)
        row["alert_red"] = cat_stats((x["cls_tgt"] >= 3).values, x["alert"].values >= 3)
        row["persistence"] = cat_stats(y, x["cls_today"].values >= 2)
        det.append(row)
    metrics["detection"] = det

    # confusion matrix: actual class vs issued alert, leads 1-3 pooled
    for tag, hs in [("lead1", [1]), ("lead3", [3]), ("leads1_3", [1, 2, 3])]:
        x = ev[ev["h"].isin(hs) & (ev["cls_tgt"] >= 0)]
        cm = pd.crosstab(x["cls_tgt"].astype(int), x["alert"].astype(int)).reindex(index=range(4), columns=range(4),
                                                                                    fill_value=0)
        metrics.setdefault("confusion", {})[tag] = cm.values.tolist()

    # reliability (P(heatwave)) for leads 1 and 3
    rel = {}
    for h in (1, 3):
        x = ev[(ev["h"] == h) & (ev["cls_tgt"] >= 0)].copy()
        x["p"] = x["p2"] + x["p3"]
        x["bin"] = pd.cut(x["p"], [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], include_lowest=True)
        g = x.groupby("bin", observed=True).agg(p_mean=("p", "mean"), obs=("cls_tgt", lambda s: (s >= 2).mean()),
                                                n=("p", "size")).reset_index(drop=True)
        rel[f"lead{h}"] = g.round(4).to_dict(orient="list")
    metrics["reliability"] = rel

    # CSI vs threshold on the validation year (how alert thresholds were chosen)
    vf = ff[(ff["date"] > pd.Timestamp(TRAIN_END)) & (ff["date"] <= pd.Timestamp(VAL_END))]
    vp = eng.predict_frame(vf)
    vp = vp[vp["h"].isin([1, 2, 3]) & (vp["cls_tgt"] >= 0)]
    P = vp[["p0", "p1", "p2", "p3"]].values
    yv = vp["cls_tgt"].values
    curves = {}
    ts = np.round(np.arange(0.05, 0.71, 0.05), 2)
    for key, pr, yy in [("yellow", P[:, 1:].sum(1), yv >= 1), ("orange", P[:, 2:].sum(1), yv >= 2),
                        ("red", P[:, 3], yv >= 3)]:
        curves[key] = [round(cat_stats(yy, pr >= t)["csi"], 3) for t in ts]
    metrics["csi_curves"] = dict(thresholds=ts.tolist(), **curves, rules=eng.rules)

    # alert performance by tier on the test period (leads 1-3 pooled)
    x = ev[ev["h"].isin([1, 2, 3]) & (ev["cls_tgt"] >= 0)]
    tiers = {}
    for lvl, name in [(1, "Yellow or higher"), (2, "Orange or higher"), (3, "Red")]:
        tiers[name] = cat_stats((x["cls_tgt"] >= lvl).values, (x["alert"] >= lvl).values)
    metrics["alert_tiers"] = tiers

    # ---------------------------------------------------------------- permutation importance (lead 3)
    h = 3
    hf = horizon_frame(ff, h, eng.normals)
    hf = hf[(hf["date"] >= TEST_START) & hf["dep_tgt"].notna() & hf[["dep_l6", "dep_m30"]].notna().all(axis=1)]
    samp = hf.sample(min(5000, len(hf)), random_state=0)
    pi = permutation_importance(eng.regs[h], samp[REG_FEATURES], samp["dep_tgt"], n_repeats=3, random_state=0,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=REG_FEATURES).sort_values(ascending=False)
    metrics["importance_lead3"] = {k: round(float(v), 4) for k, v in imp.head(15).items()}

    # ---------------------------------------------------------------- AWS validation
    aws = pd.read_parquet(OUT / "aws_hourly.parquet")
    reg = pd.read_csv(OUT / "aws_stations.csv")
    good = aws[aws["qc_flag"] == 0].copy()
    good["date"] = good["ts"].dt.floor("D")
    daily = good.groupby(["station_id", "date"]).agg(tmax=("temp_c", "max"), n=("temp_c", "size"),
                                                     hi_max=("heat_index_c", "max")).reset_index()
    daily = daily[daily["n"] >= 18].merge(reg[["station_id", "city", "type"]], on="station_id")
    cd = clean[["city", "date", "tmax"]].rename(columns={"tmax": "city_tmax"})
    daily = daily.merge(cd, on=["city", "date"])
    daily["off"] = daily["tmax"] - daily["city_tmax"]
    bytype = daily.groupby("type")["off"].agg(["mean", "std"]).round(2)
    metrics["aws_offsets"] = {k: dict(mean=float(v["mean"]), std=float(v["std"])) for k, v in bytype.iterrows()}
    metrics["aws_daily_obs"] = int(len(daily))
    cmean = daily.groupby(["city", "date"])["tmax"].mean().rename("aws_mean").reset_index()
    vv = []
    for h in (1, 3, 5):
        x = ev[(ev["h"] == h)].merge(cmean, left_on=["city", "target_date"], right_on=["city", "date"])
        err = x["tmax_pred"] - x["aws_mean"]
        err_city = x["tmax_pred"] - x["tmax_tgt"]
        vv.append(dict(h=h, n=int(len(x)), mae_vs_aws=round(float(err.abs().mean()), 3),
                       bias_vs_aws=round(float(err.mean()), 3), mae_vs_city=round(float(err_city.abs().mean()), 3)))
    metrics["aws_validation"] = vv
    # station-level heatwave agreement (IMD rules applied to AWS daily Tmax with city normal)
    nrm = ff[["city", "date", "normal"]]
    d2 = daily.merge(nrm, on=["city", "date"])
    d2["cls_aws"] = classify(d2["tmax"], d2["normal"], d2["city"].map(lambda c: CITY[c]["terrain"] == "coastal"))
    d2 = d2.merge(clean_[["city", "date", "cls"]], on=["city", "date"])
    metrics["aws_hw_days_by_type"] = d2.groupby("type").apply(
        lambda g: dict(aws=int((g["cls_aws"] >= 2).sum()), city=int((g["cls"] >= 2).sum())), include_groups=False).to_dict()
    qc = aws["qc_flag"].value_counts().to_dict()
    metrics["aws_qc"] = {str(k): int(v) for k, v in qc.items()}
    metrics["aws_rows"] = int(len(aws))
    up = aws.assign(ok=(aws["qc_flag"] != 3)).groupby("station_id")["ok"].mean()
    metrics["aws_uptime_mean"] = round(float(up.mean()), 4)
    metrics["aws_uptime_min"] = round(float(up.min()), 4)

    # ---------------------------------------------------------------- default as-of date (peak 2026 risk)
    t26 = test[(test["issue_date"] >= "2026-03-15") & (test["issue_date"] <= "2026-06-15") & test["h"].isin([1, 2, 3])]
    score = t26.groupby(["issue_date", "city"])["hri"].max().groupby("issue_date").sum()
    metrics["default_asof"] = str(score.idxmax().date())
    metrics["default_asof_score"] = round(float(score.max()), 1)

    # enriched daily table for the dashboard (normal, departure, class, heat index)
    from hw.imd import heat_index_c
    enr = clean_[["date", "city", "tmax", "tmin", "rh", "wind", "z500", "normal", "cls"]].copy()
    enr["dep"] = enr["tmax"] - enr["normal"]
    enr["hi"] = heat_index_c(enr["tmax"], enr["rh"])
    for c in ["tmax", "tmin", "rh", "wind", "z500", "normal", "dep", "hi"]:
        enr[c] = enr[c].astype("float32")
    enr["cls"] = enr["cls"].astype("int8")
    enr.to_parquet(OUT / "daily_enriched.parquet", index=False)
    json.dump({k: [round(float(x), 3) for x in v] for k, v in eng.normals.items()}, open(OUT / "normals.json", "w"))

    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1, default=str)
    print("saved metrics; default as-of:", metrics["default_asof"])
    print(json.dumps(metrics["alert_tiers"], indent=1))
    print([(d["h"], d["auc"], d["ap"], d["alert_orange_plus"]["pod"], d["alert_orange_plus"]["far"]) for d in det])
    print([(r["h"], r["model_mae"], r["persist_anomaly_mae"], r["persistence_mae"], r["coverage90"]) for r in rows[:7]])


if __name__ == "__main__":
    main()
