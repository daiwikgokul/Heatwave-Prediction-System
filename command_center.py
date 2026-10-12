import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import ALERT_LEVELS, INK, SLATE

asof = ui.setup()
la = st.session_state["lookahead"]
fc = data.forecast_at(asof)
status = data.city_status(fc, la)
n = {lvl: int((status["alert"] == lvl).sum()) for lvl in range(4)}

ui.header("Command Center",
          f"Forecast issued on the evening of {asof:%A, %d %B %Y}. Showing the worst expected day in the next "
          f"{la} day{'s' if la > 1 else ''} for each of the 10 monitored cities.")

# ------------------------------------------------------------------ instrument strip
pk = status.loc[status["peak_tmax"].idxmax()]
exposed = status.loc[status["alert"] >= 2, "pop"].sum()
aws = data.aws()
a_now = aws[(aws["date"] == asof) & (aws["ts"].dt.hour == 15) & (aws["qc_flag"] == 0)]
if len(a_now):
    danger = int((a_now["heat_index_c"] >= 41).sum())
    aws_val, aws_lab = f"{danger}", f"of {len(a_now)} AWS stations report danger-level heat index at 15:00"
else:
    aws_val, aws_lab = "n/a", "AWS network was not yet installed on this date"
first = status.loc[status["alert"] >= 2, "first_day"]
ui.readout([
    dict(val=n[3], lab="cities under Red warning", color=ALERT_LEVELS[3]["hex"]),
    dict(val=n[2], lab="cities under Orange alert", color=ALERT_LEVELS[2]["hex"]),
    dict(val=n[1], lab="cities on Yellow watch", color=ALERT_LEVELS[1]["hex"]),
    dict(val=f"{pk['peak_tmax']:.1f}°C", lab=f"peak forecast Tmax: {pk['city']}, {ui.fmt_day(asof + pd.Timedelta(days=int(pk['peak_tmax_h'])))}"),
    dict(val=f"{exposed:.0f} M", lab="people in cities at Orange or Red"),
    dict(val=aws_val, lab=aws_lab),
])

# ------------------------------------------------------------------ map + alert list
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Where is the risk?")
    fig = go.Figure()
    for lvl in range(4):
        s = status[status["alert"] == lvl]
        if s.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=s["lat"], lon=s["lon"], mode="markers+text", text=s["city"], textposition="top center",
            textfont=dict(size=12, color=INK), name=f"{ALERT_LEVELS[lvl]['colour']} · {ALERT_LEVELS[lvl]['name']}",
            marker=dict(size=16 + s["peak_hri"].clip(0, 100) * 0.32, color=ALERT_LEVELS[lvl]["hex"], opacity=0.88),
            customdata=np.column_stack([s["peak_tmax"], s["peak_dep"], s["p_hw_max"] * 100, s["peak_hri"],
                                        s["first_day"], s["region"]]),
            hovertemplate="<b>%{text}</b><br>%{customdata[5]}<br>Peak Tmax %{customdata[0]:.1f}°C "
                          "(%{customdata[1]:+.1f}°C vs normal)<br>P(heatwave) %{customdata[2]:.0f}%<br>"
                          "Risk index %{customdata[3]:.0f}/100<extra></extra>"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=21.5, lon=79.0), zoom=3.55),
                      height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(255,255,255,.85)"),
                      font=dict(family=ui.FONT))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Bubble size scales with the Heatwave Risk Index (0–100); colour is the IMD-style alert tier.")

with right:
    st.subheader("Alert board")
    rows = []
    for _, r in status.iterrows():
        when = ("no alert days" if r["alert"] == 0 else
                f"from {ui.fmt_day(asof + pd.Timedelta(days=int(r['first_day'])))} (day +{int(r['first_day'])})")
        rows.append(
            f'<div class="alert-row" style="--c:{r["colour"]}"><div class="bar"></div>'
            f'<div><div class="nm">{r["city"]}</div><div class="sm">{r["region"]}</div></div>'
            f'<div class="rt">{ui.chip(r["alert"])}<div class="sm">{r["peak_tmax"]:.1f}°C · {when}</div></div></div>')
    st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 7-day risk calendar
st.subheader("Seven-day risk calendar")
order = status["city"].tolist()
piv_a = fc.pivot(index="city", columns="h", values="alert").reindex(order)
piv_t = fc.pivot(index="city", columns="h", values="tmax_pred").reindex(order)
days = [f"{(asof + pd.Timedelta(days=int(h))):%a %d %b}" for h in piv_a.columns]
band = [[0, "#2E9E5B"], [1 / 6, "#2E9E5B"], [1 / 6, "#F2C230"], [0.5, "#F2C230"], [0.5, "#F28C28"],
        [5 / 6, "#F28C28"], [5 / 6, "#D62839"], [1, "#D62839"]]
heat = go.Figure(go.Heatmap(z=piv_a.values, x=days, y=order, zmin=0, zmax=3, colorscale=band, showscale=False,
                            xgap=3, ygap=3, hoverinfo="skip"))
for i, c in enumerate(order):
    for j in range(piv_a.shape[1]):
        lvl = int(piv_a.values[i, j])
        heat.add_annotation(x=days[j], y=c, text=f"<b>{piv_t.values[i, j]:.0f}°</b>", showarrow=False,
                            font=dict(color="#FFFFFF" if lvl in (0, 3) else INK, size=13))
heat.update_yaxes(autorange="reversed")
heat.update_xaxes(side="top")
ui.style_fig(heat, height=430, legend=False, margin=dict(l=8, r=8, t=40, b=8))
st.plotly_chart(heat, width="stretch", config={"displayModeBar": False})
st.caption("Each cell shows the forecast maximum temperature; colour shows the alert tier for that day. "
           "Forecast skill drops with lead time, so treat days 5–7 as an early heads-up rather than a firm call.")

# ------------------------------------------------------------------ monitoring context
st.subheader("Recent observations")
d = data.daily()
win = d[(d["date"] <= asof) & (d["date"] > asof - pd.Timedelta(days=45))]
n_hw = win.groupby("date")["cls"].apply(lambda s: int((s >= 2).sum())).reset_index(name="n")
nat = win.groupby("date")["dep"].mean().reset_index(name="dep")
fig2 = go.Figure()
fig2.add_bar(x=n_hw["date"], y=n_hw["n"], name="Cities in heatwave (observed)", marker_color="#E4572E", opacity=0.85)
fig2.add_scatter(x=nat["date"], y=nat["dep"], name="Mean departure from normal (°C)", yaxis="y2",
                 line=dict(color=INK, width=2.4))
fig2.update_layout(yaxis=dict(title="Cities", range=[0, 10], dtick=2),
                   yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False, zeroline=True,
                               zerolinecolor="#C5CFDC"))
ui.style_fig(fig2, height=280)
st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
