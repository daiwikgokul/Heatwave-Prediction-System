import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN

from hw import data, ui
from hw.config import ALERT_LEVELS, CITY, INK, THERMAL

asof = ui.setup()
ui.header("Hotspot map",
          "Where is heat concentrating? The heat surface is interpolated between city forecasts; clusters are found "
          "automatically from the spatial pattern of cities at Orange or Red.")

fc_all = data.forecast_at(asof)
c1, c2 = st.columns([2, 1.4])
lead = c1.select_slider("Forecast lead (days ahead)", options=list(range(1, 8)), value=min(2, 7),
                        format_func=lambda x: f"+{x} d · {ui.fmt_day(asof + pd.Timedelta(days=x))}")
metric = c2.radio("Show", ["Forecast Tmax (°C)", "Departure from normal (°C)", "Heatwave risk index"],
                  horizontal=True, label_visibility="collapsed")

f = fc_all[fc_all["h"] == lead].copy()
f["lat"] = f["city"].map(lambda c: CITY[c]["lat"])
f["lon"] = f["city"].map(lambda c: CITY[c]["lon"])
f["region"] = f["city"].map(lambda c: CITY[c]["region"])
f["pop"] = f["city"].map(lambda c: CITY[c]["pop"])
f["dep"] = f["tmax_pred"] - f["normal_tgt"]
f["p_hw"] = (f["p2"] + f["p3"]) * 100
col, unit = {"Forecast Tmax (°C)": ("tmax_pred", "°C"), "Departure from normal (°C)": ("dep", "°C"),
             "Heatwave risk index": ("hri", "")}[metric]

# ------------------------------------------------------------------ inverse-distance-weighted surface
glat, glon = np.meshgrid(np.linspace(8, 33, 60), np.linspace(68, 90, 60), indexing="ij")
def idw(vals):
    d2 = (glat[..., None] - f["lat"].values) ** 2 + ((glon[..., None] - f["lon"].values) * 0.9) ** 2
    w = 1.0 / np.maximum(d2, 0.04) ** 1.5
    return (w * vals).sum(-1) / w.sum(-1)
surf = idw(f[col].values)
# fade the surface with distance to the nearest station so it does not claim knowledge far from the network
dmin = np.sqrt(np.min((glat[..., None] - f["lat"].values) ** 2 + ((glon[..., None] - f["lon"].values) * 0.9) ** 2, axis=-1))
mask = dmin < 5.5

fig = go.Figure()
fig.add_trace(go.Densitymap(lat=glat[mask].ravel(), lon=glon[mask].ravel(), z=surf[mask].ravel(), radius=26,
                            colorscale=[[i / (len(THERMAL) - 1), c] for i, c in enumerate(THERMAL)],
                            opacity=0.62, zmin=float(f[col].min()) - 0.5, zmax=float(f[col].max()) + 0.5,
                            colorbar=dict(title=unit or "0–100", thickness=12, len=0.7), hoverinfo="skip"))
fig.add_trace(go.Scattermap(
    lat=f["lat"], lon=f["lon"], mode="markers+text", text=f["city"], textposition="top right",
    textfont=dict(size=12, color=INK),
    marker=dict(size=13, color=[ALERT_LEVELS[int(a)]["hex"] for a in f["alert"]], opacity=1),
    customdata=np.column_stack([f["tmax_pred"], f["dep"], f["p_hw"], f["hri"]]),
    hovertemplate="<b>%{text}</b><br>Tmax %{customdata[0]:.1f}°C (%{customdata[1]:+.1f})<br>"
                  "P(heatwave) %{customdata[2]:.0f}%<br>Risk %{customdata[3]:.0f}<extra></extra>", showlegend=False))
fig.update_layout(map=dict(style="carto-positron", center=dict(lat=21.0, lon=79.0), zoom=3.6), height=540,
                  margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family=ui.FONT))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Dots show the alert tier of each city on the chosen day. The coloured surface is interpolated only "
           "within ~500 km of a monitored city.")

# ------------------------------------------------------------------ spatial clusters (DBSCAN on haversine distance)
st.subheader("Automatically detected hotspot clusters")
hot = f[f["alert"] >= 2].copy()
if len(hot) >= 2:
    coords = np.radians(hot[["lat", "lon"]].values)
    lab = DBSCAN(eps=700 / 6371.0, min_samples=2, metric="haversine").fit_predict(coords)
    hot["cluster"] = lab
else:
    hot["cluster"] = -1
groups = [(k, g) for k, g in hot.groupby("cluster") if k >= 0]
if groups:
    cols = st.columns(min(len(groups), 3))
    for i, (k, g) in enumerate(groups):
        col_ = cols[i % len(cols)]
        worst = int(g["alert"].max())
        col_.markdown(
            f'<div class="advisory" style="--c:{ALERT_LEVELS[worst]["hex"]}"><h4>Cluster {chr(65 + i)}: '
            f'{len(g)} cities</h4><div>{", ".join(g["city"])}</div><div class="note" style="margin-top:6px">'
            f'{g["region"].nunique()} region(s) · {g["pop"].sum():.0f} M people · peak Tmax {g["tmax_pred"].max():.1f}°C '
            f'· highest risk index {g["hri"].max():.0f}</div></div>', unsafe_allow_html=True)
    singles = hot[hot["cluster"] == -1]
else:
    singles = hot
if len(singles):
    st.markdown(f"**Isolated hot spots:** {', '.join(singles['city'])}")
if hot.empty:
    st.success("No city is forecast at Orange or Red on this day, so no hotspot clusters are flagged.")

# ------------------------------------------------------------------ ranking table
st.subheader("Ranking for the selected day")
tbl = f.sort_values("hri", ascending=False)[["city", "region", "tmax_pred", "dep", "p_hw", "hri", "alert"]].copy()
tbl["alert"] = tbl["alert"].map(lambda a: ALERT_LEVELS[int(a)]["colour"])
tbl.columns = ["City", "Region", "Forecast Tmax (°C)", "vs normal (°C)", "P(heatwave) %", "Risk index", "Alert"]
st.dataframe(tbl, hide_index=True, width="stretch", column_config={
    "Forecast Tmax (°C)": st.column_config.NumberColumn(format="%.1f"),
    "vs normal (°C)": st.column_config.NumberColumn(format="%+.1f"),
    "P(heatwave) %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    "Risk index": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
})

reg = f.groupby("region").agg(cities=("city", lambda s: ", ".join(s)), worst_alert=("alert", "max"),
                              peak_tmax=("tmax_pred", "max"), mean_hri=("hri", "mean")).reset_index()
reg["worst_alert"] = reg["worst_alert"].map(lambda a: ALERT_LEVELS[int(a)]["colour"])
reg.columns = ["Region", "Cities", "Worst alert", "Peak Tmax (°C)", "Mean risk index"]
st.markdown("**By region**")
st.dataframe(reg.sort_values("Mean risk index", ascending=False), hide_index=True, width="stretch",
             column_config={"Peak Tmax (°C)": st.column_config.NumberColumn(format="%.1f"),
                            "Mean risk index": st.column_config.NumberColumn(format="%.0f")})
