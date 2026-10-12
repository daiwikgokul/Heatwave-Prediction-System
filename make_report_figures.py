"""Figures for the project report -> assets/report/*.png"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from hw.config import ASSETS, DATA, INK, THERMAL

OUT = ASSETS / "report"; OUT.mkdir(parents=True, exist_ok=True)
P = DATA / "processed"
M = json.load(open(P / "metrics.json"))
d = pd.read_parquet(P / "daily_enriched.parquet"); d["year"] = d["date"].dt.year
pred = pd.read_parquet(P / "predictions.parquet")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8FA3B8",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK})
RED, ORG = "#D62839", "#E4572E"

# 1 heatwave days per year
by = d.groupby("year")["cls"].agg(hw=lambda s: int((s == 2).sum()), sev=lambda s: int((s == 3).sum()))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(by.index, by["hw"], color="#F28C28", label="Heatwave"); ax.bar(by.index, by["sev"], bottom=by["hw"], color=RED, label="Severe heatwave")
ax.set_ylabel("City-days per year (10 cities)"); ax.legend(frameon=False); ax.set_title("Heatwave days in the synthetic archive", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig1_heatwave_days.png", dpi=170); plt.close()

# 2 MAE vs lead
r = pd.DataFrame(M["regression"]); r = r[r["scope"] == "All seasons"]
fig, ax = plt.subplots(figsize=(8, 3.6))
for col, lab, c, ls in [("model_mae", "Model", ORG, "-"), ("persist_anomaly_mae", "Persistence of today's anomaly", INK, "--"),
                        ("persistence_mae", "Persistence of today's Tmax", "#8FA3B8", ":"), ("climatology_mae", "Climatology", "#C5CFDC", ":")]:
    ax.plot(r["h"], r[col], ls, color=c, lw=2.6 if col == "model_mae" else 1.8, marker="o", ms=4, label=lab)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("MAE (°C)"); ax.set_ylim(0, 2); ax.legend(frameon=False, fontsize=9)
ax.set_title("Tmax forecast error on the hold-out period", loc="left"); fig.tight_layout(); fig.savefig(OUT / "fig2_mae_lead.png", dpi=170); plt.close()

# 3 detection skill
det = M["detection"]; h = [x["h"] for x in det]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(h, [x["alert_orange_plus"]["csi"] for x in det], color=ORG, lw=2.6, marker="o", label="Model (Orange or higher)")
ax.plot(h, [x["persistence"]["csi"] for x in det], "--", color=INK, lw=1.8, marker="o", ms=4, label="Persistence")
ax.plot(h, [x["brier_skill"] for x in det], ":", color="#7A8696", lw=1.8, marker="o", ms=4, label="Brier skill vs climatology")
ax.axhline(0, color="#C5CFDC", lw=1); ax.set_xlabel("Lead time (days)"); ax.set_ylabel("CSI / skill score"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Heatwave detection skill by lead time", loc="left"); fig.tight_layout(); fig.savefig(OUT / "fig3_detection.png", dpi=170); plt.close()

# 4 case study
p = pred[(pred["city"] == "Ahmedabad") & (pred["h"] == 3) & (pred["target_date"].between("2026-04-15", "2026-06-10"))].sort_values("target_date")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.fill_between(p["target_date"], p["tmax_lo"], p["tmax_hi"], color=ORG, alpha=0.15, label="90% range")
ax.plot(p["target_date"], p["tmax_tgt"], color=INK, lw=2, label="Observed Tmax"); ax.plot(p["target_date"], p["tmax_pred"], color=ORG, lw=2, label="Forecast made 3 days earlier")
hw = p[p["cls_tgt"] >= 2]; ax.scatter(hw["target_date"], hw["tmax_tgt"], color="#8E1B2E", marker="D", s=22, zorder=5, label="Actual heatwave day")
al = p[p["alert"] >= 2]; ax.scatter(al["target_date"], [p["tmax_pred"].min() - 1] * len(al), color="#F28C28", marker="^", s=22, label="Orange/Red alert issued")
ax.set_ylabel("°C"); ax.legend(frameon=False, fontsize=8, ncol=2); ax.set_title("Case study: Ahmedabad, spring 2026", loc="left")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "fig4_case_study.png", dpi=170); plt.close()
print("figures saved", sorted(x.name for x in OUT.iterdir()))
