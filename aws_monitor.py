import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import AWS_START, CITY_NAMES, INK, THERMAL

asof = ui.setup()
ui.header("AWS monitoring",
          "Automated Weather Stations give street-level readings. They show where the city average hides "
          "hotter neighbourhoods, and they check the forecast against the ground.")

if asof < pd.Timestamp(AWS_START):
    st.info(f"The AWS network starts on {pd.Timestamp(AWS_START):%d %b %Y}. Choose a later forecast issue date in the sidebar.")
    st.stop()

reg = data.stations()
aws = data.aws()
c1, c2 = st.columns([1, 1.6])
city = c1.selectbox("City", ["All cities"] + CITY_NAMES)
hour = c2.slider("Hour of day", 0, 23, 15, format="%d:00")

day = aws[aws["date"] == asof].merge(reg[["station_id", "city", "name", "type", "lat", "lon"]], on="station_id")
if city != "All cities":
    day = day[day["city"] == city]
cur = day[day["ts"].dt.hour == hour]
ok = cur[cur["qc_flag"] == 0]

ui.readout([
    dict(val=f"{len(ok)}/{len(cur)}", lab=f"stations reporting valid data at {hour}:00"),
    dict(val=f"{ok['temp_c'].max():.1f}°C" if len(ok) else "n/a",
         lab=f"hottest reading: {ok.loc[ok['temp_c'].idxmax(), 'name']}, {ok.loc[ok['temp_c'].idxmax(), 'city']}" if len(ok) else "no data"),
    dict(val=int((ok["heat_index_c"] >= 41).sum()), lab="stations in danger-level heat index (41°C or more)"),
    dict(val=int((ok["battery_v"] < 3.6).sum()), lab="stations with low battery (under 3.6 V)"),
    dict(val=int((day["qc_flag"] > 0).sum()), lab="readings flagged by quality control today"),
])

# ------------------------------------------------------------------ map + table
a, b = st.columns([1.4, 1], gap="large")
with a:
    m = cur.copy()
    m["val"] = m["heat_index_c"]
    fig = go.Figure()
    sym = {"Urban Core": "circle", "Industrial": "square", "Suburban": "circle", "Green Belt": "circle"}
    good = m[m["qc_flag"] == 0]
    fig.add_trace(go.Scattermap(
        lat=good["lat"], lon=good["lon"], mode="markers", showlegend=False,
        marker=dict(size=14, color=good["val"], colorscale=[[i / (len(THERMAL) - 1), c] for i, c in enumerate(THERMAL)],
                    cmin=26, cmax=54, colorbar=dict(title="Heat index °C", thickness=12, len=0.7), opacity=0.95),
        customdata=np.column_stack([good["name"], good["city"], good["type"], good["temp_c"], good["rh_pct"],
                                    good["heat_index_c"]]),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>Temp %{customdata[3]:.1f}°C · "
                      "RH %{customdata[4]:.0f}%<br>Heat index %{customdata[5]:.1f}°C<extra></extra>"))
    bad = m[m["qc_flag"] != 0]
    if len(bad):
        fig.add_trace(go.Scattermap(lat=bad["lat"], lon=bad["lon"], mode="markers", name="No valid reading",
                                    marker=dict(size=12, color="#7A8696", opacity=0.9), text=bad["name"],
                                    hovertemplate="<b>%{text}</b><br>offline or flagged<extra></extra>"))
    zoom, ctr = (3.5, dict(lat=21.5, lon=79.0)) if city == "All cities" else (
        10.2, dict(lat=reg[reg["city"] == city]["lat"].mean(), lon=reg[reg["city"] == city]["lon"].mean()))
    fig.update_layout(map=dict(style="carto-positron", center=ctr, zoom=zoom), height=470, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=0.02, bgcolor="rgba(255,255,255,.8)"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with b:
    st.markdown("**Station readings**")
    t = cur[["name", "city", "type", "temp_c", "rh_pct", "heat_index_c", "battery_v", "rssi_dbm", "qc_flag"]].copy()
    t["status"] = np.where(t["qc_flag"] == 3, "Offline", np.where(t["qc_flag"] > 0, "Flagged", "Online"))
    t = t.drop(columns="qc_flag").sort_values("heat_index_c", ascending=False)
    t.columns = ["Station", "City", "Type", "Temp °C", "RH %", "Heat idx °C", "Battery V", "Signal dBm", "Status"]
    st.dataframe(t, hide_index=True, width="stretch", height=470, column_config={
        c: st.column_config.NumberColumn(format="%.1f") for c in ["Temp °C", "RH %", "Heat idx °C", "Battery V", "Signal dBm"]})

# ------------------------------------------------------------------ local heat islands
st.subheader("Neighbourhood differences")
d = data.daily()
dd = aws[(aws["date"] <= asof) & (aws["date"] > asof - pd.Timedelta(days=60)) & (aws["qc_flag"] == 0)]
dm = dd.groupby(["station_id", "date"]).agg(tmax=("temp_c", "max"), n=("temp_c", "size")).reset_index()
dm = dm[dm["n"] >= 18].merge(reg[["station_id", "city", "type"]], on="station_id").merge(
    d[["city", "date", "tmax", "dep"]].rename(columns={"tmax": "city_tmax"}), on=["city", "date"])
dm["offset"] = dm["tmax"] - dm["city_tmax"]
if city != "All cities":
    dm = dm[dm["city"] == city]
dm["Day type"] = np.where(dm["dep"] >= 3, "Hot spell (3°C+ above normal)", "Ordinary day")
if len(dm):
    fig = px.box(dm, x="type", y="offset", color="Day type", category_orders={"type": list(reg["type"].unique())},
                 color_discrete_map={"Ordinary day": "#8FC3D9", "Hot spell (3°C+ above normal)": "#E4572E"})
    fig.update_layout(yaxis_title="Station Tmax minus city Tmax (°C)", xaxis_title="", legend_title="")
    fig.add_hline(y=0, line_color="#97A6B8")
    ui.style_fig(fig, height=330)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Last 60 days. Dense built-up sites run hotter than the city reading, and the gap widens in hot spells.")

# ------------------------------------------------------------------ forecast validation
st.subheader("Forecast versus ground truth")
pred = data.predictions()
tdays = pd.date_range(asof - pd.Timedelta(days=13), asof, freq="D")
cmean = dm.groupby(["city", "date"])["tmax"].mean().reset_index() if city != "All cities" else \
    dm.groupby(["city", "date"])["tmax"].mean().reset_index()
vc = city if city != "All cities" else "Delhi"
if city == "All cities":
    st.caption("Showing Delhi. Pick a city above to switch.")
obs_s = dd[dd["station_id"].isin(reg[reg["city"] == vc]["station_id"])].groupby(["date"])["temp_c"].max()
p1 = pred[(pred["city"] == vc) & (pred["h"] == 1) & pred["target_date"].isin(tdays)].set_index("target_date")
p3 = pred[(pred["city"] == vc) & (pred["h"] == 3) & pred["target_date"].isin(tdays)].set_index("target_date")
cityobs = d[(d["city"] == vc) & d["date"].isin(tdays)].set_index("date")["tmax"]
fig = go.Figure()
fig.add_scatter(x=cityobs.index, y=cityobs.values, name="City observed Tmax", line=dict(color=INK, width=2.5))
for nm, ts, col in [("AWS Tmax (hottest station)", obs_s, "#8E1B2E")]:
    ts = ts[ts.index.isin(tdays)]
    fig.add_scatter(x=ts.index, y=ts.values, name=nm, mode="lines+markers", line=dict(color=col, width=2, dash="dot"))
fig.add_scatter(x=p1.index, y=p1["tmax_pred"], name="Forecast made 1 day earlier", line=dict(color="#E4572E", width=2.5))
fig.add_scatter(x=p3.index, y=p3["tmax_pred"], name="Forecast made 3 days earlier", line=dict(color="#F6B35A", width=2.5))
fig.update_yaxes(title="°C")
fig.update_layout(title=f"{vc}: last 14 days")
ui.style_fig(fig, height=340)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
mv = data.metrics()["aws_validation"]
st.dataframe(pd.DataFrame(mv).rename(columns={"h": "Lead (d)", "n": "City-days", "mae_vs_aws": "MAE vs AWS mean (°C)",
                                               "bias_vs_aws": "Bias vs AWS mean (°C)", "mae_vs_city": "MAE vs city obs (°C)"}),
             hide_index=True, width="stretch")
st.caption("Hold-out validation over all cities since the stations were installed. The negative bias against AWS reflects "
           "the stations' warmer urban siting.")

# ------------------------------------------------------------------ device health
st.subheader("Device health")
h = aws[(aws["date"] <= asof) & (aws["date"] > asof - pd.Timedelta(days=30))].merge(reg[["station_id", "city", "name"]], on="station_id")
if city != "All cities":
    h = h[h["city"] == city]
health = h.groupby(["station_id", "name", "city"]).agg(
    uptime=("qc_flag", lambda s: 100 * (s != 3).mean()), batt_min=("battery_v", "min"),
    rssi=("rssi_dbm", "mean"), flags=("qc_flag", lambda s: int(((s == 1) | (s == 2)).sum()))).reset_index()
health.columns = ["ID", "Station", "City", "Uptime % (30 d)", "Lowest battery V", "Mean signal dBm", "QC flags"]
st.dataframe(health.sort_values("Uptime % (30 d)"), hide_index=True, width="stretch", column_config={
    "Uptime % (30 d)": st.column_config.ProgressColumn(min_value=80, max_value=100, format="%.1f%%"),
    "Lowest battery V": st.column_config.NumberColumn(format="%.2f"),
    "Mean signal dBm": st.column_config.NumberColumn(format="%.0f")})
