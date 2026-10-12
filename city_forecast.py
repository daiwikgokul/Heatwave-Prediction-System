import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import ALERT_LEVELS, CITY, CITY_NAMES, INK, SEV_COLOURS, SEVERITY, SLATE
from hw.models import required_tmax

asof = ui.setup()
la = st.session_state["lookahead"]
fc_all = data.forecast_at(asof)
status = data.city_status(fc_all, la)
if "city_sel" not in st.session_state:
    st.session_state["city_sel"] = status.iloc[0]["city"]

c_sel, c_rev = st.columns([2, 1.4])
city = c_sel.selectbox("City", CITY_NAMES, key="city_sel")
reveal = c_rev.toggle("Reveal what actually happened (verification mode)", value=False,
                      help="Overlays the observed Tmax and heatwave class for the forecast days. "
                           "Only possible for dates that already have observations.")

ui.header(f"{city} forecast", f"{CITY[city]['state']} · {CITY[city]['region']} · "
                              f"{'coastal' if CITY[city]['terrain'] == 'coastal' else 'plains'} IMD criteria · "
                              f"issued {asof:%d %b %Y}")

fc = fc_all[fc_all["city"] == city].sort_values("h").reset_index(drop=True)
st_row = status[status["city"] == city].iloc[0]
obs = data.observed_on(asof, city).iloc[0]
coastal = CITY[city]["terrain"] == "coastal"
fc["date"] = [asof + pd.Timedelta(days=int(h)) for h in fc["h"]]

ui.readout([
    dict(val=ui.chip(st_row["alert"]).replace("font-size:.78rem", ""), lab=f"alert tier (next {la} days)",
         color=st_row["colour"]),
    dict(val=f"{obs['tmax']:.1f}°C", lab=f"observed Tmax today ({obs['dep']:+.1f}°C vs normal {obs['normal']:.1f}°C)"),
    dict(val=f"{st_row['peak_tmax']:.1f}°C", lab=f"peak forecast Tmax, {ui.fmt_day(asof + pd.Timedelta(days=int(st_row['peak_tmax_h'])))}"),
    dict(val=f"{st_row['p_hw_max'] * 100:.0f}%", lab="highest chance of heatwave in window"),
    dict(val=f"{st_row['peak_hri']:.0f}", lab="peak Heatwave Risk Index (0–100)"),
])

# ------------------------------------------------------------------ chart: observed + forecast + thresholds
d = data.daily()
hist = d[(d["city"] == city) & (d["date"] <= asof) & (d["date"] > asof - pd.Timedelta(days=21))]
hw_thr_hist = required_tmax(hist["normal"].values, np.full(len(hist), coastal))[1]
hw_thr_fc = required_tmax(fc["normal_tgt"].values, np.full(len(fc), coastal))[1]

fig = go.Figure()
fig.add_scatter(x=list(fc["date"]) + list(fc["date"])[::-1], y=list(fc["tmax_hi"]) + list(fc["tmax_lo"])[::-1],
                fill="toself", fillcolor="rgba(228,87,46,0.16)", line=dict(width=0), name="90% forecast range",
                hoverinfo="skip")
fig.add_scatter(x=hist["date"], y=hist["normal"], name="Normal (climatology)", line=dict(color="#8FA3B8", width=2, dash="dot"))
fig.add_scatter(x=list(hist["date"]) + list(fc["date"]), y=list(hw_thr_hist) + list(hw_thr_fc),
                name="IMD heatwave threshold", line=dict(color="#8E1B2E", width=1.6, dash="dash"))
fig.add_scatter(x=hist["date"], y=hist["tmax"], name="Observed Tmax", mode="lines+markers",
                line=dict(color=INK, width=2.6), marker=dict(size=5))
fig.add_scatter(x=[asof] + list(fc["date"]), y=[obs["tmax"]] + list(fc["tmax_pred"]), name="Forecast Tmax",
                mode="lines+markers", line=dict(color="#E4572E", width=3), marker=dict(size=8))
if reveal:
    act = fc.dropna(subset=["tmax_tgt"])
    fig.add_scatter(x=act["date"], y=act["tmax_tgt"], name="Actually observed", mode="markers",
                    marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.4), symbol="diamond"))
fig.add_vline(x=asof, line_width=1, line_color="#97A6B8")
fig.add_annotation(x=asof, y=1, yref="paper", text="issue date", showarrow=False, xanchor="right", yanchor="bottom",
                   font=dict(color=SLATE, size=11))
fig.update_yaxes(title="°C")
ui.style_fig(fig, height=400)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------------ day cards
cols = st.columns(7)
for (_, r), col in zip(fc.iterrows(), cols):
    extra = ""
    if reveal and not np.isnan(r["tmax_tgt"]):
        extra = (f'<div class="sm" style="margin-top:4px">Observed {r["tmax_tgt"]:.1f}°C<br>'
                 f'<b>{SEVERITY[int(r["cls_tgt"])]}</b></div>')
    col.markdown(
        f'<div class="panel" style="padding:10px 12px;border-top:5px solid {ALERT_LEVELS[int(r["alert"])]["hex"]}">'
        f'<div class="note">{ui.fmt_day(r["date"])} · +{int(r["h"])}d</div>'
        f'<div style="font-family:{ui.DISPLAY};font-size:1.7rem;font-weight:700;line-height:1.15">{r["tmax_pred"]:.1f}°</div>'
        f'<div class="note">{r["tmax_lo"]:.0f}–{r["tmax_hi"]:.0f}° · {r["dep_pred"]:+.1f} vs normal</div>'
        f'<div style="margin:6px 0 2px">{ui.chip(r["alert"], ALERT_LEVELS[int(r["alert"])]["colour"])}</div>'
        f'<div class="note">P(heatwave) {(r["p2"] + r["p3"]) * 100:.0f}%</div>{extra}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ probabilities
st.subheader("How likely is each severity?")
fig2 = go.Figure()
for k in range(4):
    fig2.add_bar(x=[ui.fmt_day(x) for x in fc["date"]], y=fc[f"p{k}"] * 100, name=SEVERITY[k], marker_color=SEV_COLOURS[k],
                 hovertemplate="%{y:.1f}%<extra>" + SEVERITY[k] + "</extra>")
fig2.update_layout(barmode="stack", yaxis=dict(title="Probability (%)", range=[0, 100]))
ui.style_fig(fig2, height=300)
st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
st.caption("Probabilities blend a gradient-boosting severity classifier with the regression forecast and its "
           "error spread. Heatwave = IMD criteria (Tmax ≥ 40 °C and ≥ 4.5 °C above normal on plains; "
           "≥ 37 °C on the coast; or Tmax ≥ 45 °C).")

# ------------------------------------------------------------------ what-if simulator
st.subheader("What-if simulator")
st.write("Stress-test the model: nudge recent observations in this city and see how the forecast and alert tier respond.")
with st.form("whatif"):
    c1, c2, c3, c4 = st.columns(4)
    d_t = c1.slider("Extra warming of recent Tmax (°C)", -4.0, 6.0, 2.0, 0.5)
    d_rh = c2.slider("Change in afternoon humidity (%)", -20, 20, -5, 1)
    d_z = c3.slider("Stronger upper-air ridge (m)", -30, 80, 30, 5,
                    help="Synthetic 500-hPa height anomaly. Ridges usually precede heatwaves by several days.")
    days = c4.slider("Applied to the last N days", 1, 7, 3)
    go_btn = st.form_submit_button("Run scenario", type="primary")
if go_btn:
    with st.spinner("Re-running the models ..."):
        sc = data.engine().forecast(d, asof, perturb=dict(city=city, d_tmax=d_t, d_rh=d_rh, d_z500=d_z, days=days))
    sc = sc[sc["city"] == city].sort_values("h")
    fig3 = go.Figure()
    fig3.add_scatter(x=fc["h"], y=fc["hri"], name="Baseline", mode="lines+markers", line=dict(color=INK, width=2.5))
    fig3.add_scatter(x=sc["h"], y=sc["hri"], name="Scenario", mode="lines+markers", line=dict(color="#E4572E", width=3))
    fig3.update_xaxes(title="Lead time (days)", dtick=1)
    fig3.update_yaxes(title="Heatwave Risk Index")
    ui.style_fig(fig3, height=300)
    a, b = st.columns([1.3, 1])
    a.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
    tab = pd.DataFrame({"Lead (d)": sc["h"].values, "Baseline Tmax": fc["tmax_pred"].round(1).values,
                        "Scenario Tmax": sc["tmax_pred"].round(1).values,
                        "Baseline alert": [ALERT_LEVELS[int(x)]["colour"] for x in fc["alert"]],
                        "Scenario alert": [ALERT_LEVELS[int(x)]["colour"] for x in sc["alert"]]})
    b.dataframe(tab, hide_index=True, width="stretch")
    worst_b, worst_s = int(fc["alert"].max()), int(sc["alert"].max())
    st.info(f"Worst alert tier over the 7 days moves from **{ALERT_LEVELS[worst_b]['colour']}** to "
            f"**{ALERT_LEVELS[worst_s]['colour']}** under this scenario.")
