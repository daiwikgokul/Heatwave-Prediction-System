import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import CITY, CITY_NAMES, INK, REGIONS, SEASONS, SEV_COLOURS, THERMAL

ui.setup()
ui.header("Regional analytics",
          "How do temperatures and heatwaves behave across the 7 climatic regions and 4 IMD seasons? "
          "This is the Phase II analysis behind the models (2005 to 2026).")

d = data.daily()
f1, f2 = st.columns([2, 1])
regs = f1.multiselect("Regions", REGIONS, default=REGIONS)
yr = f2.slider("Years", 2005, 2026, (2005, 2026))
cities = [c for c in CITY_NAMES if CITY[c]["region"] in regs]
if not cities:
    st.warning("Select at least one region.")
    st.stop()
sel = d[d["city"].isin(cities) & d["year"].between(*yr)].copy()
sel["hw"] = (sel["cls"] >= 2).astype(int)

tabs = st.tabs(["Heatwave frequency", "Seasonal patterns", "Spells", "Long-term trend"])

# ------------------------------------------------------------------ 1 frequency
with tabs[0]:
    by = sel.groupby("year").agg(hot=("cls", lambda s: int((s == 1).sum())), hw=("cls", lambda s: int((s == 2).sum())),
                                 sev=("cls", lambda s: int((s == 3).sum()))).reset_index()
    ui.readout([dict(val=int(by["hw"].sum() + by["sev"].sum()), lab="heatwave city-days in selection"),
                dict(val=int(by["sev"].sum()), lab="of which severe"),
                dict(val=f"{(by['hw'] + by['sev']).max()}", lab=f"most in one year ({int(by.loc[(by['hw'] + by['sev']).idxmax(), 'year'])})"),
                dict(val=f"{sel[sel['hw'] == 1]['city'].nunique()}/{len(cities)}", lab="cities with at least one heatwave day")])
    fig = go.Figure()
    for k, (nm, c) in enumerate([("hot", SEV_COLOURS[1]), ("hw", SEV_COLOURS[2]), ("sev", SEV_COLOURS[3])]):
        fig.add_bar(x=by["year"], y=by[nm], name=["Hot day", "Heatwave", "Severe heatwave"][k], marker_color=c)
    fig.update_layout(barmode="stack", yaxis_title="City-days per year")
    ui.style_fig(fig, height=330)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    piv = sel.pivot_table(index="city", columns="year", values="hw", aggfunc="sum").reindex(cities)
    hm = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, xgap=2, ygap=2,
                              colorscale=[[i / (len(THERMAL) - 1), c] for i, c in enumerate(THERMAL)][1:] or THERMAL,
                              colorbar=dict(title="days", thickness=10),
                              hovertemplate="%{y} %{x}: %{z} heatwave days<extra></extra>"))
    hm.update_yaxes(autorange="reversed")
    hm.update_layout(title="Heatwave days per city and year")
    ui.style_fig(hm, height=360, legend=False)
    st.plotly_chart(hm, width="stretch", config={"displayModeBar": False})

# ------------------------------------------------------------------ 2 seasonal
with tabs[1]:
    mon = sel.groupby(["city", "month"])["tmax"].mean().reset_index()
    mon["m"] = pd.to_datetime(mon["month"], format="%m").dt.strftime("%b")
    fig = px.line(mon, x="month", y="tmax", color="city", markers=False,
                  color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Set2)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.update_layout(title="Mean daily maximum temperature by month", yaxis_title="°C", legend_title="")
    for lo, hi, nm in [(0.5, 2.5, "Winter"), (2.5, 5.5, "Pre-monsoon"), (5.5, 9.5, "Monsoon"), (9.5, 12.5, "Post-monsoon")]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#FFFFFF" if nm in ("Winter", "Monsoon") else "#EEF2F7", opacity=0.55,
                      line_width=0, annotation_text=nm, annotation_position="top left",
                      annotation_font=dict(size=11, color="#5B6B7C"))
    ui.style_fig(fig, height=380)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    seasons_order = list(SEASONS)
    bx = px.box(sel, x="season", y="tmax", color="region", category_orders={"season": seasons_order},
                color_discrete_sequence=px.colors.qualitative.Safe)
    bx.update_layout(title="Distribution of daily Tmax by season and region", yaxis_title="°C", xaxis_title="",
                     legend_title="")
    bx.update_traces(marker=dict(size=3))
    ui.style_fig(bx, height=400)
    st.plotly_chart(bx, width="stretch", config={"displayModeBar": False})

    sc = sel[sel["hw"] == 1].groupby(["month"]).size().reindex(range(1, 13), fill_value=0)
    st.caption("Heatwave city-days by calendar month: " +
               ", ".join(f"{pd.Timestamp(2000, m, 1):%b} {int(v)}" for m, v in sc.items() if v > 0) + ".")

# ------------------------------------------------------------------ 3 spells
with tabs[2]:
    sp = data.heatwave_spells(sel)
    if sp.empty:
        st.info("No heatwave spells (2 or more consecutive days) in this selection.")
    else:
        ui.readout([dict(val=len(sp), lab="heatwave spells (2+ days)"), dict(val=int(sp["days"].max()), lab="longest spell (days)"),
                    dict(val=f"{sp['days'].mean():.1f}", lab="average length (days)"),
                    dict(val=f"{sp['peak_tmax'].max():.1f}°C", lab="highest Tmax in any spell")])
        a, b = st.columns([1, 1.2])
        h = px.histogram(sp, x="days", nbins=int(sp["days"].max()), color_discrete_sequence=["#E4572E"])
        h.update_layout(title="Spell length", xaxis_title="Days", yaxis_title="Spells", bargap=0.08)
        ui.style_fig(h, height=320, legend=False)
        a.plotly_chart(h, width="stretch", config={"displayModeBar": False})
        top = sp.sort_values(["days", "peak_tmax"], ascending=False).head(10).copy()
        top["start"] = top["start"].dt.strftime("%d %b %Y")
        top["end"] = top["end"].dt.strftime("%d %b %Y")
        top.columns = ["City", "Start", "End", "Days", "Peak Tmax (°C)", "Severe days"]
        b.markdown("**Ten longest spells**")
        b.dataframe(top, hide_index=True, width="stretch",
                    column_config={"Peak Tmax (°C)": st.column_config.NumberColumn(format="%.1f")})
        by_city = sp.groupby("city").agg(spells=("days", "size"), longest=("days", "max"),
                                         days=("days", "sum")).reindex(cities).fillna(0).astype(int).reset_index()
        st.markdown("**Spells per city**")
        st.dataframe(by_city.rename(columns={"city": "City", "spells": "Spells", "longest": "Longest (d)", "days": "Total days"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4 trend
with tabs[3]:
    pre = d[d["city"].isin(cities) & d["month"].isin([3, 4, 5])].groupby(["city", "year"])["tmax"].mean().reset_index()
    pre = pre[pre["year"].between(*yr)]
    fig = px.line(pre, x="year", y="tmax", color="city", markers=True,
                  color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Set2)
    fig.update_layout(title="Pre-monsoon (Mar–May) mean Tmax", yaxis_title="°C", legend_title="")
    ui.style_fig(fig, height=360)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    rows = []
    hwy = d[d["city"].isin(cities)].groupby(["city", "year"])["cls"].apply(lambda s: int((s >= 2).sum())).reset_index(name="hw")
    for c in cities:
        p = pre[pre["city"] == c]
        h = hwy[(hwy["city"] == c) & hwy["year"].between(*yr)]
        if len(p) >= 4:
            s1 = np.polyfit(p["year"], p["tmax"], 1)[0] * 10
            s2 = np.polyfit(h["year"], h["hw"], 1)[0] * 10
            e1 = h[h["year"] <= 2014]["hw"].mean()
            e2 = h[h["year"] >= 2015]["hw"].mean()
            rows.append(dict(City=c, **{"Pre-monsoon warming (°C/decade)": s1, "Heatwave days/yr 2005–14": e1,
                                        "Heatwave days/yr 2015–26": e2, "Change (days/yr per decade)": s2}))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Pre-monsoon warming (°C/decade)": st.column_config.NumberColumn(format="%+.2f"),
            "Heatwave days/yr 2005–14": st.column_config.NumberColumn(format="%.1f"),
            "Heatwave days/yr 2015–26": st.column_config.NumberColumn(format="%.1f"),
            "Change (days/yr per decade)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("The synthetic archive was generated with a warming trend of 0.035 °C/yr and rising event frequency, "
               "so the upward trend seen here is by construction. With real IMD data, the same analysis would "
               "quantify the observed trend.")
