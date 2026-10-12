# Intelligent Heatwave Risk Prediction and Early Warning System

Use case KJS-CES-01 (Climate Intelligence for Heatwave Monitoring, Prediction, and Early Warning).
All data are **synthetic**: for teaching and demonstration only.

## Run it
```bash
pip install -r requirements.txt
python scripts/build_all.py        # generate data, clean it, simulate AWS feed, train models (~2 min)
python scripts/evaluate.py         # back-test, metrics, AWS validation (~1 min)
streamlit run app.py
```
The `data/` and `models/` folders already contain the outputs of these steps, so you can run `streamlit run app.py` straight away.
Optional: `python scripts/make_report_figures.py` rebuilds the report charts.

## Pages
Command Center, City Forecast (what-if simulator, verification mode), Hotspot Map, Regional Analytics, AWS Monitoring,
Advisories & Warnings (human approval log), Model Lab, Data & Governance.

## Layout
- `hw/` data generator, IMD rules, features, models, advisory engine, UI helpers
- `views/` one file per page; `app.py` entry point
- `scripts/` build, evaluate, figures

## Optional LLM advisories
Set `ANTHROPIC_API_KEY` (and optionally `HEATWAVE_LLM_MODEL`) before starting Streamlit to enable the LLM drafting toggle.
Outputs are checked against the forecast facts. This path could not be tested without a key; the template engine is the default.
