import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hw import data, ui
from hw.config import CITIES, CITY_NAMES, INK

ui.setup()
ui.header("Data and governance",
          "Where the numbers come from, how they were cleaned, and how the system meets the responsible-AI "
          "expectations of the use case.")
st.warning("All data in this application are synthetic. They were generated to behave like Indian city climates and "
           "are not IMD observations. Nothing here should be used for real warnings.")

tabs = st.tabs(["Pipeline", "Data explorer", "Dictionary", "Responsible AI"])
qa = data.quality()
M = data.metrics()

with tabs[0]:
    st.markdown("**Five phases, as in the use case**")
    steps = [("I", "Acquire and clean", "Synthetic IMD-style archive, quality checks, repair"),
             ("II", "Analyse and predict", "Regional and seasonal analysis; Tmax, heatwave and severity models"),
             ("III", "Monitor locally", "40 simulated AWS stations, hourly, with health telemetry"),
             ("IV", "Validate", "Forecasts checked against the city record and AWS readings"),
             ("V", "Warn and advise", "Alert tiers, drafted advisories, human approval, log")]
    cols = st.columns(5)
    for c, (n, t, s) in zip(cols, steps):
        c.markdown(f'<div class="panel" style="min-height:130px"><div style="font-family:{ui.DISPLAY};font-weight:700">'
                   f'Phase {n}</div><b>{t}</b><div class="note" style="margin-top:4px">{s}</div></div>', unsafe_allow_html=True)
    st.write("")
    ui.readout([dict(val=f"{qa['rows_raw']:,}", lab="rows in the raw archive"),
                dict(val=qa["duplicates_removed"], lab="duplicate rows removed"),
                dict(val=qa["missing_days_added"], lab="missing city-days re-created"),
                dict(val=qa["sentinel_99_9"], lab="99.9 missing-value codes found"),
                dict(val=qa["spikes_flagged"], lab="isolated Tmax spikes flagged"),
                dict(val=f"{qa['repair_mae_tmax']} °C", lab="average error on repaired Tmax values")])
    st.caption("The cleaning step is checked against the hidden truth because the data are synthetic. "
               f"{qa['pct_cells_repaired']}% of all values needed repair. Short gaps (up to 5 days) are interpolated in time; "
               "longer gaps fall back to the day-of-year average.")
    raw, clean = data.raw_daily(), data.daily()
    c1, c2 = st.columns([1, 2])
    cs = c1.selectbox("Show raw vs cleaned", CITY_NAMES, index=0)
    rw = raw[(raw["city"] == cs) & (raw["date"].between("2019-04-01", "2019-06-30"))]
    cl = clean[(clean["city"] == cs) & (clean["date"].between("2019-04-01", "2019-06-30"))]
    fig = go.Figure()
    fig.add_scatter(x=rw["date"], y=rw["tmax"], mode="markers", name="Raw archive", marker=dict(color="#8E1B2E", size=6))
    fig.add_scatter(x=cl["date"], y=cl["tmax"], name="Cleaned", line=dict(color=INK, width=2))
    fig.update_yaxes(title="Tmax (°C)", range=[20, 55])
    ui.style_fig(fig, height=320)
    c2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Raw points at 99.9 sit above the range shown. Spring 2019 shown for the chosen city.")

with tabs[1]:
    d = data.daily()
    c1, c2, c3 = st.columns([1, 1, 1.4])
    cs = c1.multiselect("Cities", CITY_NAMES, default=["Delhi"])
    rng = c2.date_input("Dates", value=(pd.Timestamp("2026-04-01").date(), pd.Timestamp("2026-06-30").date()))
    only = c3.checkbox("Heatwave days only")
    x = d[d["city"].isin(cs)]
    if isinstance(rng, tuple) and len(rng) == 2:
        x = x[x["date"].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))]
    if only:
        x = x[x["cls"] >= 2]
    x = x[["date", "city", "region", "tmax", "tmin", "rh", "wind", "z500", "normal", "dep", "hi", "cls"]]
    st.dataframe(x, hide_index=True, width="stretch", height=380)
    st.download_button("Download selection (CSV)", x.to_csv(index=False), "heatwave_daily_selection.csv", "text/csv")
    st.markdown("**Monitored cities and stations**")
    st.dataframe(pd.DataFrame(CITIES)[["name", "state", "region", "lat", "lon", "elev", "terrain", "pop"]].rename(
        columns={"name": "City", "state": "State", "region": "Region", "lat": "Lat", "lon": "Lon", "elev": "Elevation (m)",
                 "terrain": "IMD criteria", "pop": "Population (M, approx.)"}), hide_index=True, width="stretch")

with tabs[2]:
    st.dataframe(pd.DataFrame([
        ("date, city", "Observation day and city (10 cities, 7 climatic regions)"),
        ("tmax, tmin", "Daily maximum and minimum temperature, °C"),
        ("rh", "Afternoon relative humidity, %"),
        ("wind", "Daily mean wind speed, km/h"),
        ("z500", "Upper-air ridge index in metres: a synthetic stand-in for the 500-hPa height anomaly, the kind of "
                 "large-scale signal that precedes real heatwaves"),
        ("normal, dep", "Smoothed day-of-year normal (2005–2022) and departure from it, °C"),
        ("hi", "Heat index from Tmax and afternoon humidity (Rothfusz regression), °C"),
        ("cls", "IMD-style class: 0 normal, 1 hot day, 2 heatwave, 3 severe heatwave"),
        ("AWS: temp_c, rh_pct, wind_kmh, heat_index_c", "Hourly station readings"),
        ("AWS: battery_v, rssi_dbm", "Device health: battery voltage and radio signal"),
        ("AWS: qc_flag", "0 ok, 1 out of range, 2 spike, 3 missing"),
    ], columns=["Field", "Meaning"]), hide_index=True, width="stretch")
    st.markdown("**Heatwave rules used** (IMD-style): plains need Tmax of at least 40 °C and 4.5 °C above normal "
                "(severe at 6.5 °C), or Tmax of at least 45 °C (severe at 47 °C). Coastal stations need 37 °C and the same "
                "departures. A hot day is the same Tmax floor with a 3 °C departure. IMD declares a heatwave after two "
                "consecutive days; this system classifies single days and reports spells separately.")

with tabs[3]:
    st.dataframe(pd.DataFrame([
        ("Privacy and data protection", "Low", "No personal data. The decision log stores reviewer names only; access to the log file should be restricted."),
        ("AI security and trustworthiness", "High", "Input checks (range, spike, missing), validation against AWS, back-tested skill by lead time shown in Model Lab, quality-control flags on every sensor reading."),
        ("Responsible AI", "High", "Probabilities and 90% ranges are shown, not single numbers; confidence is labelled by lead time; known weaknesses (cool bias on extremes, no skill by day 7) are stated on screen."),
        ("Generative AI / LLMs", "High", "Advisory drafts come from templates by default. The optional LLM mode is restricted to a facts pack and every number is checked against it."),
        ("Agentic AI", "Low", "Not used. The system does not send anything on its own."),
        ("Human in the loop", "High", "No advisory counts as sent until a named reviewer approves it, with channels and a note; all decisions are logged."),
        ("Regulatory and governance", "Medium", "Alert tiers follow IMD colour codes. Real use would need IMD sign-off on thresholds and messaging."),
    ], columns=["Area (from the use case)", "Relevance", "How this application addresses it"]), hide_index=True, width="stretch")
    st.markdown("**Known limits**")
    st.markdown(
        "- Synthetic data: skill numbers describe this simulation, not real-world performance.\n"
        "- The ridge index was built to lead heatwaves by 3 to 8 days. Real forecasts would need numerical weather model output.\n"
        "- Heatwave probability has no skill over climatology by day 7; Tmax skill over persistence is modest at 1 day.\n"
        "- On heatwave days the model forecasts too cool, and about 11–14% of observed values fall outside the stated 90% range.\n"
        "- AWS sites are illustrative locations, not real stations.")
