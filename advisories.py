import datetime as dt
import json

import pandas as pd
import streamlit as st

from hw import advisory as adv
from hw import data, ui
from hw.config import ALERT_LEVELS, AWS_START, CITY_NAMES, DATA

asof = ui.setup()
la = st.session_state["lookahead"]
fc = data.forecast_at(asof)
status = data.city_status(fc, la)
LOG = DATA / "processed" / "dissemination_log.csv"
if "dlog" not in st.session_state:
    st.session_state["dlog"] = pd.read_csv(LOG).to_dict("records") if LOG.exists() else []

ui.header("Advisories and warnings",
          "Forecasts become plain-language advice for each audience. Every draft needs a human reviewer's approval "
          "before it counts as sent.")

c1, c2, c3 = st.columns([1.2, 1.4, 1])
default_city = status.iloc[0]["city"]
city = c1.selectbox("City", CITY_NAMES, index=CITY_NAMES.index(st.session_state.get("city_sel", default_city)))
stake = c2.selectbox("Audience", adv.STAKEHOLDERS)
lang_name = c3.selectbox("Language", list(adv.LANGS), disabled=(stake != "Citizens"),
                         help="Translations are available for the citizen advisory.")
lang = adv.LANGS[lang_name] if stake == "Citizens" else "en"

aws_hot = None
if asof >= pd.Timestamp(AWS_START):
    a = data.aws()
    reg = data.stations()
    x = a[(a["date"] == asof) & (a["qc_flag"] == 0)].merge(reg[["station_id", "city", "name"]], on="station_id")
    x = x[x["city"] == city]
    if len(x):
        aws_hot = x.loc[x["temp_c"].idxmax()].to_dict()

facts = adv.build_facts(city, fc, asof, aws_hot, la)
lvl = facts["alert_level"]
a1, a2 = st.columns([1.5, 1], gap="large")

with a2:
    st.markdown(f"**Facts the advisory may use**  \n{ui.chip(lvl)}", unsafe_allow_html=True)
    st.json(facts, expanded=False)
    use_llm = st.toggle("Draft with an LLM (Claude API)", value=False, disabled=not adv.llm_available(),
                        help="Needs the ANTHROPIC_API_KEY environment variable. The output is checked against the facts.")
    if not adv.llm_available():
        st.caption("LLM drafting is off because no ANTHROPIC_API_KEY is set. The template engine is used.")

draft_key = f"{city}|{stake}|{lang}|{asof}|{use_llm}"
if st.session_state.get("draft_key") != draft_key:
    text, engine = None, "Template engine"
    if use_llm:
        try:
            text = adv.llm_advisory(stake, facts, lang)
            engine = "LLM (Claude)"
        except Exception as e:  # fall back safely
            st.warning(f"LLM call failed ({type(e).__name__}); using the template engine instead.")
    if text is None:
        text = adv.render_md(adv.template_advisory(stake, facts, lang))
    st.session_state.update(draft_key=draft_key, draft_text=text, draft_engine=engine)

with a1:
    st.markdown(f'<div class="advisory" style="--c:{ALERT_LEVELS[lvl]["hex"]}">'
                f'<div class="note">DRAFT · {st.session_state["draft_engine"]} · {stake}</div></div>', unsafe_allow_html=True)
    edited = st.text_area("Advisory text (edit before approving)", value=st.session_state["draft_text"], height=330,
                          key=f"ta_{draft_key}")
    chk = adv.consistency_check(edited, facts)
    if chk["ok"]:
        st.success(f"Consistency check passed: all {chk['checked']} numbers trace back to the forecast facts.")
    else:
        st.warning("Numbers not found in the forecast facts: " + ", ".join(chk["unverified"]) +
                   ". Check them before approving.")

st.subheader("Reviewer decision")
with st.form("review"):
    r1, r2, r3 = st.columns([1, 1.4, 1])
    who = r1.text_input("Reviewer name", placeholder="e.g. duty meteorologist")
    chans = r2.multiselect("Channels", ["SMS", "WhatsApp", "Email", "Mobile app push", "Radio and TV", "Loudspeaker announcements"],
                           default=["SMS", "Mobile app push"])
    decision = r3.radio("Decision", ["Approve and send", "Hold", "Reject"], horizontal=False)
    note = st.text_input("Reviewer note (optional)")
    sub = st.form_submit_button("Record decision", type="primary")
if sub:
    if not who.strip():
        st.error("Enter the reviewer's name. Every decision is logged against a person.")
    elif decision == "Approve and send" and not chans:
        st.error("Choose at least one channel.")
    else:
        rec = dict(time=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), issue_date=asof.strftime("%Y-%m-%d"), city=city,
                   audience=stake, language=lang_name, alert=facts["alert_colour"], engine=st.session_state["draft_engine"],
                   decision=decision, channels=", ".join(chans) if decision == "Approve and send" else "",
                   reviewer=who.strip(), note=note, consistency_ok=chk["ok"], text=edited.replace("\n", " "))
        st.session_state["dlog"].append(rec)
        try:
            pd.DataFrame(st.session_state["dlog"]).to_csv(LOG, index=False)
        except Exception:
            pass
        st.success(f"{decision}: recorded for {city} ({stake}).")

st.subheader("Decision log")
if st.session_state["dlog"]:
    log = pd.DataFrame(st.session_state["dlog"]).drop(columns=["text"]).iloc[::-1]
    st.dataframe(log, hide_index=True, width="stretch")
    st.download_button("Download full log (CSV)", pd.DataFrame(st.session_state["dlog"]).to_csv(index=False),
                       file_name="dissemination_log.csv", mime="text/csv")
else:
    st.caption("No decisions recorded yet.")

# ------------------------------------------------------------------ bulk
st.subheader("Cities needing a decision now")
need = status[status["alert"] >= 1][["city", "label", "first_day", "peak_tmax", "p_hw_max"]].copy()
if need.empty:
    st.caption("No city is at Yellow or above in this window.")
else:
    need["p_hw_max"] = (need["p_hw_max"] * 100).round(0)
    need.columns = ["City", "Alert", "First alert day (+d)", "Peak Tmax (°C)", "Peak P(heatwave) %"]
    st.dataframe(need, hide_index=True, width="stretch")
