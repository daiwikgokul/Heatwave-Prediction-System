import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import ALERT_LEVELS, CITY_NAMES, INK, SEVERITY

ui.setup()
M = data.metrics()
ui.header("Model lab",
          "How good are the forecasts? Everything here is measured on 2024 onwards, a period the models never saw in "
          "training, and compared with simple baselines.")
ds = M["dataset"]
ui.readout([dict(val=f"{ds['rows']:,}", lab="city-days in the archive"),
            dict(val=f"{ds['test_days']:,}", lab="hold-out forecast days tested"),
            dict(val=ds["test_hw_city_days"], lab="actual heatwave city-days in the hold-out"),
            dict(val="8", lab="models: 7 Tmax regressors + 1 severity classifier")])

tabs = st.tabs(["Temperature forecast", "Heatwave detection", "Alert tiers", "What drives it", "Case study"])
reg = pd.DataFrame(M["regression"])

with tabs[0]:
    scope = st.radio("Season scope", reg["scope"].unique().tolist(), horizontal=True)
    r = reg[reg["scope"] == scope]
    fig = go.Figure()
    for nm, col, c, dash in [("Model", "model_mae", "#E4572E", None), ("Persistence of today's anomaly", "persist_anomaly_mae", INK, "dash"),
                             ("Persistence of today's Tmax", "persistence_mae", "#8FA3B8", "dot"),
                             ("Climatology", "climatology_mae", "#C5CFDC", "dot")]:
        fig.add_scatter(x=r["h"], y=r[col], name=nm, mode="lines+markers", line=dict(color=c, width=3 if nm == "Model" else 2, dash=dash))
    fig.update_xaxes(title="Lead time (days)", dtick=1)
    fig.update_yaxes(title="Mean absolute error (°C)", rangemode="tozero")
    ui.style_fig(fig, height=360)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    t = r[["h", "n", "model_mae", "model_rmse", "bias", "r2", "persist_anomaly_mae", "skill_vs_persist_anom", "coverage90"]].copy()
    t.columns = ["Lead (d)", "Samples", "MAE °C", "RMSE °C", "Bias °C", "R²", "Persistence MAE °C", "Skill vs persistence", "90% range coverage"]
    t["Skill vs persistence"] = t["Skill vs persistence"] * 100
    t["90% range coverage"] = t["90% range coverage"] * 100
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "Skill vs persistence": st.column_config.NumberColumn(format="%.0f%%", help="1 - MAE/MAE_persistence"),
        "90% range coverage": st.column_config.NumberColumn(format="%.0f%%")})
    hw = pd.DataFrame(M["regression_hw_days"])
    st.markdown("**On days that really were heatwaves**, the model runs cool, as most regression models do on extremes:")
    st.dataframe(hw.rename(columns={"h": "Lead (d)", "n": "Heatwave city-days", "mae": "MAE °C", "bias": "Bias °C"}),
                 hide_index=True, width="stretch")

det = pd.DataFrame(M["detection"])
with tabs[1]:
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=det["h"], y=det["auc"], name="Blend (used)", line=dict(color="#E4572E", width=3), mode="lines+markers")
    fig.add_scatter(x=det["h"], y=det["auc_classifier"], name="Classifier only", line=dict(color="#8FC3D9", width=2), mode="lines+markers")
    fig.add_scatter(x=det["h"], y=det["auc_regression"], name="Regression only", line=dict(color=INK, width=2, dash="dash"), mode="lines+markers")
    fig.update_layout(title="Ranking skill (ROC AUC)")
    fig.update_xaxes(title="Lead (days)", dtick=1)
    fig.update_yaxes(range=[0.8, 1.0])
    ui.style_fig(fig, height=330)
    c1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    fig = go.Figure()
    fig.add_scatter(x=det["h"], y=[d["alert_orange_plus"]["csi"] for d in M["detection"]], name="Model (Orange or higher)",
                    line=dict(color="#E4572E", width=3), mode="lines+markers")
    fig.add_scatter(x=det["h"], y=[d["persistence"]["csi"] for d in M["detection"]], name="Persistence",
                    line=dict(color=INK, width=2, dash="dash"), mode="lines+markers")
    fig.add_scatter(x=det["h"], y=det["brier_skill"], name="Brier skill vs climatology", line=dict(color="#7A8696", width=2, dash="dot"),
                    mode="lines+markers")
    fig.update_layout(title="Hit rate: Critical Success Index")
    fig.update_xaxes(title="Lead (days)", dtick=1)
    ui.style_fig(fig, height=330)
    c2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    tb = pd.DataFrame({"Lead (d)": det["h"], "POD": [d["alert_orange_plus"]["pod"] for d in M["detection"]],
                       "False alarm ratio": [d["alert_orange_plus"]["far"] for d in M["detection"]],
                       "CSI": [d["alert_orange_plus"]["csi"] for d in M["detection"]],
                       "Persistence CSI": [d["persistence"]["csi"] for d in M["detection"]],
                       "Brier skill": det["brier_skill"]})
    st.dataframe(tb, hide_index=True, width="stretch")
    st.caption("POD = share of real heatwave days that triggered Orange or Red. False alarm ratio = share of alerts that "
               "were not heatwaves. Skill fades with lead time; by day 6–7 the model is no better than climatology.")
    rel = M["reliability"]
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#C5CFDC", dash="dash"), name="Perfect", hoverinfo="skip")
    for k, c in [("lead1", "#E4572E"), ("lead3", INK)]:
        fig.add_scatter(x=rel[k]["p_mean"], y=rel[k]["obs"], mode="lines+markers", name=f"Lead {k[-1]} d", line=dict(color=c))
    fig.update_layout(title="Reliability of heatwave probability", xaxis_title="Forecast probability", yaxis_title="Observed frequency")
    ui.style_fig(fig, height=330)
    c1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    cm = np.array(M["confusion"]["leads1_3"])
    labels = [ALERT_LEVELS[i]["colour"] for i in range(4)]
    fig = go.Figure(go.Heatmap(z=np.log10(cm + 1), x=labels, y=SEVERITY, text=cm, texttemplate="%{text}", colorscale="Blues",
                               showscale=False, xgap=2, ygap=2))
    fig.update_layout(title="Actual class vs alert issued (leads 1–3)", xaxis_title="Alert issued", yaxis_title="What happened")
    fig.update_yaxes(autorange="reversed")
    ui.style_fig(fig, height=330, legend=False)
    c2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with tabs[2]:
    cv = M["csi_curves"]
    fig = go.Figure()
    for k, c in [("yellow", "#F2C230"), ("orange", "#F28C28"), ("red", "#D62839")]:
        fig.add_scatter(x=cv["thresholds"], y=cv[k], name=k.title(), line=dict(color=c, width=3), mode="lines+markers")
    for k, c in [("yellow_p_any", "#F2C230"), ("orange_p_hw", "#F28C28"), ("red_p_severe", "#D62839")]:
        fig.add_vline(x=cv["rules"][k], line_color=c, line_dash="dot")
    fig.update_layout(title="Choosing alert thresholds on the 2023 validation year",
                      xaxis_title="Probability threshold", yaxis_title="CSI")
    ui.style_fig(fig, height=340)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Dotted lines mark the thresholds in use: Yellow at {cv['rules']['yellow_p_any']:.0%} (hot day or worse), "
               f"Orange at {cv['rules']['orange_p_hw']:.0%} (heatwave or worse), Red at {cv['rules']['red_p_severe']:.0%} (severe). "
               "They were read off the flat top of these curves and rounded.")
    tt = pd.DataFrame(M["alert_tiers"]).T[["pod", "far", "csi", "freq_bias", "tp", "fp", "fn"]].reset_index()
    tt.columns = ["Tier (leads 1–3, hold-out)", "POD", "False alarm ratio", "CSI", "Frequency bias", "Hits", "False alarms", "Misses"]
    st.dataframe(tt, hide_index=True, width="stretch")

with tabs[3]:
    imp = pd.Series(M["importance_lead3"]).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#E4572E"))
    fig.update_layout(title="Permutation importance, 3-day Tmax model", xaxis_title="Increase in MAE (°C) when shuffled")
    ui.style_fig(fig, height=460, legend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Names: dep_l0 = today's departure from normal; z500_l0 = today's upper-air ridge index; rel_dep = today's anomaly "
               "relative to neighbouring cities; nat_max_z500 = strongest ridge anywhere in the network; nbr_z500 = ridge "
               "strength at neighbouring cities. Today's anomaly dominates; the ridge matters mostly in the rare big events.")

with tabs[4]:
    pred = data.predictions()
    d = data.daily()
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("City", CITY_NAMES, index=CITY_NAMES.index("Nagpur"))
    year = c2.selectbox("Year", [2024, 2025, 2026], index=2)
    lead = c3.select_slider("Forecast lead (days)", options=list(range(1, 8)), value=3)
    p = pred[(pred["city"] == city) & (pred["h"] == lead) & (pred["target_date"].dt.year == year) &
             (pred["target_date"].dt.month.isin([3, 4, 5, 6]))].sort_values("target_date")
    fig = go.Figure()
    fig.add_scatter(x=list(p["target_date"]) + list(p["target_date"])[::-1], y=list(p["tmax_hi"]) + list(p["tmax_lo"])[::-1],
                    fill="toself", fillcolor="rgba(228,87,46,0.15)", line=dict(width=0), name="90% range", hoverinfo="skip")
    fig.add_scatter(x=p["target_date"], y=p["tmax_tgt"], name="Observed", line=dict(color=INK, width=2.4))
    fig.add_scatter(x=p["target_date"], y=p["tmax_pred"], name=f"Forecast made {lead} d earlier", line=dict(color="#E4572E", width=2.4))
    hits = p[p["cls_tgt"] >= 2]
    fig.add_scatter(x=hits["target_date"], y=hits["tmax_tgt"], mode="markers", name="Actual heatwave day",
                    marker=dict(size=9, color="#8E1B2E", symbol="diamond"))
    al = p[p["alert"] >= 2]
    fig.add_scatter(x=al["target_date"], y=[p["tmax_pred"].min() - 1] * len(al), mode="markers", name="Orange/Red alert issued",
                    marker=dict(size=8, color="#F28C28", symbol="triangle-up"))
    fig.update_yaxes(title="°C")
    ui.style_fig(fig, height=380)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("March to June. Triangles along the bottom mark days when an Orange or Red alert had been issued for that date.")
