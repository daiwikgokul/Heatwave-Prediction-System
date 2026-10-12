"""Phase II - AI models: Tmax regressors (1-7 day lead) + severity classifier + risk logic."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import (ALERT_LEVELS, ALERT_RULES, CITY, HORIZONS, HRI_WEIGHTS, MODELS, MONTH_TO_SEASON,
                     SEASONS, SEVERITY, TRAIN_END, VAL_END)
from .features import CLF_FEATURES, REG_FEATURES, build_feature_frame, horizon_frame
from .imd import base_threshold, classify

ARTIFACT = MODELS / "heatwave_models.joblib"


# --------------------------------------------------------------------------- risk logic
def hri(p: np.ndarray) -> np.ndarray:
    """Heatwave Risk Index 0-100 from class probabilities (n,4)."""
    return 100.0 * (np.asarray(p) @ np.asarray(HRI_WEIGHTS))


def alert_from_probs(p: np.ndarray, rules: dict | None = None) -> np.ndarray:
    """IMD-style colour code (0 green .. 3 red) from class probabilities (n,4)."""
    r = rules or ALERT_RULES
    p = np.atleast_2d(p)
    p3 = p[:, 3]
    p23 = p[:, 2] + p[:, 3]
    p123 = p[:, 1] + p[:, 2] + p[:, 3]
    lvl = np.zeros(len(p), dtype=int)
    lvl[p123 >= r["yellow_p_any"]] = 1
    lvl[p23 >= r["orange_p_hw"]] = 2
    lvl[p3 >= r["red_p_severe"]] = 3
    return lvl


def required_tmax(normal, coastal):
    """Tmax needed to reach class >=1, >=2, >=3 on a given day (IMD rules, see imd.classify)."""
    normal = np.asarray(normal, dtype=float)
    coastal = np.asarray(coastal, dtype=bool)
    base = base_threshold(coastal)
    cap = np.where(coastal, np.inf, 1.0)
    t1 = np.minimum(np.maximum(base, normal + 3.0), np.where(coastal, np.inf, 45.0))
    t2 = np.minimum(np.maximum(base, normal + 4.5), np.where(coastal, np.inf, 45.0))
    t3 = np.minimum(np.maximum(base, normal + 6.5), np.where(coastal, np.inf, 47.0))
    return t1, np.maximum(t1, t2), np.maximum(np.maximum(t1, t2), t3)


def gaussian_cumulative(normal, coastal, tmax_pred, sigma) -> np.ndarray:
    """P(class>=1), P(class>=2), P(class>=3) from the regression forecast and its error spread."""
    t1, t2, t3 = required_tmax(normal, coastal)
    sigma = np.maximum(sigma, 0.3)
    return np.column_stack([norm.sf((t - tmax_pred) / sigma) for t in (t1, t2, t3)])


def blend(pc: np.ndarray, pg_cum: np.ndarray, w_reg: float = 0.5) -> np.ndarray:
    """Average the classifier's and the regression-based cumulative probabilities -> class probs (n,4)."""
    pc_cum = np.column_stack([pc[:, 1:].sum(1), pc[:, 2:].sum(1), pc[:, 3]])
    cum = (1 - w_reg) * pc_cum + w_reg * pg_cum
    cum = np.minimum.accumulate(cum, axis=1)            # enforce P(>=1) >= P(>=2) >= P(>=3)
    p = np.zeros((len(cum), 4))
    p[:, 3] = cum[:, 2]
    p[:, 2] = cum[:, 1] - cum[:, 2]
    p[:, 1] = cum[:, 0] - cum[:, 1]
    p[:, 0] = 1 - cum[:, 0]
    return p


# --------------------------------------------------------------------------- training
def _train_mask(hf: pd.DataFrame, end: str, start: str | None = None):
    m = hf["target_date"] <= pd.Timestamp(end)
    if start:
        m &= hf["date"] > pd.Timestamp(start)
    return m


def fit(clean: pd.DataFrame, normals: dict, verbose: bool = True) -> dict:
    ff = build_feature_frame(clean, normals)
    frames = {h: horizon_frame(ff, h, normals) for h in HORIZONS}

    regs, resid_rows = {}, []
    for h in HORIZONS:
        hf = frames[h]
        tr = hf[_train_mask(hf, TRAIN_END)].dropna(subset=["dep_tgt"] + ["dep_l6", "dep_m30"])
        va = hf[_train_mask(hf, VAL_END, TRAIN_END)].dropna(subset=["dep_tgt", "dep_l6", "dep_m30"])
        m = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.06, max_leaf_nodes=24,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=h)
        m.fit(tr[REG_FEATURES], tr["dep_tgt"])
        pred = m.predict(va[REG_FEATURES])
        res = va["dep_tgt"].values - pred
        resid_rows.append(pd.DataFrame({"h": h, "season": va["target_date"].dt.month.map(MONTH_TO_SEASON).values,
                                        "res": res}))
        regs[h] = m
        if verbose:
            print(f"  regressor h={h}: train n={len(tr):,}  val MAE={np.abs(res).mean():.3f}", flush=True)

    # empirical 90% prediction intervals per horizon x season (validation residuals)
    resid = pd.concat(resid_rows)
    intervals = {}
    for (h, s), g in resid.groupby(["h", "season"]):
        intervals.setdefault(int(h), {})[s] = (float(g["res"].quantile(0.05)), float(g["res"].quantile(0.95)))

    # stacked-horizon severity classifier
    stack = pd.concat([frames[h] for h in HORIZONS], ignore_index=True)
    tr = stack[_train_mask(stack, TRAIN_END)].dropna(subset=["cls_tgt", "dep_l6", "dep_m30"])
    tr = tr[tr["cls_tgt"] >= 0]
    if verbose:
        print(f"  classifier: train rows={len(tr):,}  class counts={tr['cls_tgt'].value_counts().sort_index().to_dict()}",
              flush=True)
    clf = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, max_leaf_nodes=24,
                                         min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    clf.fit(tr[CLF_FEATURES], tr["cls_tgt"].astype(int))

    art = {"regs": regs, "clf": clf, "normals": normals, "intervals": intervals, "rules": dict(ALERT_RULES),
           "reg_features": REG_FEATURES, "clf_features": CLF_FEATURES}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(art, ARTIFACT, compress=3)
    return art


# --------------------------------------------------------------------------- inference
class Engine:
    """Loads the trained artefacts and produces forecasts from a history frame."""

    def __init__(self, path=ARTIFACT):
        a = joblib.load(path)
        self.regs, self.clf = a["regs"], a["clf"]
        self.normals, self.intervals, self.rules = a["normals"], a["intervals"], a["rules"]

    # batch: forecasts for *every* row of the feature frame (used for back-testing)
    def predict_frame(self, ff: pd.DataFrame, horizons=HORIZONS, with_target=True) -> pd.DataFrame:
        outs = []
        for h in horizons:
            hf = horizon_frame(ff, h, self.normals, with_target=with_target)
            ok = hf[["dep_l6", "dep_m30"]].notna().all(axis=1)
            hf = hf[ok].copy()
            dep_hat = self.regs[h].predict(hf[REG_FEATURES])
            hf["tmax_pred"] = hf["normal_tgt"].values + dep_hat
            season = hf["target_date"].dt.month.map(MONTH_TO_SEASON)
            lo = np.array([self.intervals[h][s][0] for s in season])
            hi_ = np.array([self.intervals[h][s][1] for s in season])
            hf["tmax_lo"] = hf["tmax_pred"] + lo
            hf["tmax_hi"] = hf["tmax_pred"] + hi_
            sigma = (hf["tmax_hi"] - hf["tmax_lo"]).values / 3.29
            pg = gaussian_cumulative(hf["normal_tgt"].values, hf["coastal"].values.astype(bool),
                                     hf["tmax_pred"].values, sigma)
            pc = self._proba(hf[CLF_FEATURES])
            hf["pc_ge2"], hf["pg_ge2"] = pc[:, 2:].sum(1), pg[:, 1]
            P = blend(pc, pg)
            for k in range(4):
                hf[f"p{k}"] = P[:, k]
            hf["hri"] = hri(P)
            hf["alert"] = alert_from_probs(P, self.rules)
            hf["pred_class"] = P.argmax(1)
            outs.append(hf)
        return pd.concat(outs, ignore_index=True)

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(X)
        P = np.zeros((len(X), 4))
        for j, c in enumerate(self.clf.classes_):
            P[:, int(c)] = raw[:, j]
        return P

    def forecast(self, history: pd.DataFrame, asof, perturb: dict | None = None) -> pd.DataFrame:
        """7-day forecast for all cities, issued at the end of `asof`.

        perturb: optional what-if {city, d_tmax, d_rh, d_z500, days} applied to recent observations.
        """
        asof = pd.Timestamp(asof)
        h = history[(history["date"] <= asof) & (history["date"] > asof - pd.Timedelta(days=90))].copy()
        if perturb and perturb.get("city"):
            m = (h["city"] == perturb["city"]) & (h["date"] > asof - pd.Timedelta(days=int(perturb.get("days", 3))))
            h.loc[m, "tmax"] = h.loc[m, "tmax"] + perturb.get("d_tmax", 0.0)
            h.loc[m, "tmin"] = h.loc[m, "tmin"] + 0.6 * perturb.get("d_tmax", 0.0)
            h.loc[m, "rh"] = (h.loc[m, "rh"] + perturb.get("d_rh", 0.0)).clip(5, 100)
            h.loc[m, "z500"] = h.loc[m, "z500"] + perturb.get("d_z500", 0.0)
        ff = build_feature_frame(h, self.normals)
        ff = ff[ff["date"] == asof]
        out = self.predict_frame(ff, with_target=False)
        out["issue_date"] = asof
        return out.sort_values(["city", "h"]).reset_index(drop=True)
