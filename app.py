"""Intelligent Heatwave Risk Prediction & Early Warning System  (use case KJS-CES-01)

Run:  streamlit run app.py
"""
import datetime as dt

import streamlit as st

from hw import data, ui
from hw.config import DATA_END

st.set_page_config(page_title="Heatwave Early Warning System", page_icon=":material/thermostat:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

if not data.ready():
    st.error("Data and models have not been built yet.")
    st.markdown("Run these two commands once from the project folder, then refresh this page:")
    st.code("python scripts/build_all.py\npython scripts/evaluate.py", language="bash")
    st.stop()

ui.setup()
MIN_D, MAX_D = dt.date(2024, 1, 1), dt.date.fromisoformat(DATA_END)


def _jump_peak():
    st.session_state["asof"] = ui.default_asof().date()


def _jump_latest():
    st.session_state["asof"] = MAX_D


with st.sidebar:
    st.markdown('<div class="brand">Heatwave Early Warning<small>Risk prediction for 10 Indian cities · '
                'KJS-CES-01</small></div>', unsafe_allow_html=True)
    st.write("")
    st.date_input("Forecast issue date", min_value=MIN_D, max_value=MAX_D, key="asof",
                  help="The system behaves as if it were this evening: it only sees data up to this date. "
                       "Dates after 2023 were never seen during model training.")
    c1, c2 = st.columns(2)
    c1.button("Peak 2026 heat", on_click=_jump_peak, width="stretch")
    c2.button("Latest", on_click=_jump_latest, width="stretch")
    st.slider("Look-ahead window (days)", 1, 7, key="lookahead",
              help="Warnings consider the worst forecast day inside this window.")
    st.divider()
    st.caption("All data are synthetic: generated for teaching and demonstration. "
               "Do not use for real-world warnings.")

pages = [
    st.Page("views/command_center.py", title="Command Center", icon=":material/radar:", default=True),
    st.Page("views/city_forecast.py", title="City Forecast", icon=":material/query_stats:"),
    st.Page("views/hotspots.py", title="Hotspot Map", icon=":material/map:"),
    st.Page("views/analytics.py", title="Regional Analytics", icon=":material/monitoring:"),
    st.Page("views/aws_monitor.py", title="AWS Monitoring", icon=":material/sensors:"),
    st.Page("views/advisories.py", title="Advisories & Warnings", icon=":material/campaign:"),
    st.Page("views/model_lab.py", title="Model Lab", icon=":material/science:"),
    st.Page("views/data_governance.py", title="Data & Governance", icon=":material/database:"),
]
st.navigation(pages).run()
