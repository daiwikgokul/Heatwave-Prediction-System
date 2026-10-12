"""Run once:  python scripts/build_all.py
Generates the synthetic database, cleans it, simulates AWS data, trains the models and
writes all artefacts the Streamlit app needs (data/, models/)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hw import models
from hw.config import DATA, MODELS, TRAIN_END
from hw.datagen import generate_aws, generate_clean, make_raw, station_registry
from hw.imd import compute_normals
from hw.preprocess import clean_daily, qc_aws


def main():
    t0 = time.time()
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    print("[1/5] generating synthetic archive ...", flush=True)
    truth = generate_clean()
    raw = make_raw(truth)
    raw.to_csv(DATA / "raw" / "synthetic_imd_daily_raw.csv", index=False)

    print("[2/5] cleaning (Phase I) ...", flush=True)
    clean, qa = clean_daily(raw)
    # we know the truth, so we can measure how well the repair worked
    m = clean.merge(truth[["date", "city", "tmax"]], on=["date", "city"], suffixes=("", "_true"))
    fixed = m[(m["tmax"] - m["tmax_true"]).abs() > 1e-6]
    qa["repaired_tmax_cells"] = int(len(fixed))
    qa["repair_mae_tmax"] = round(float((fixed["tmax"] - fixed["tmax_true"]).abs().mean()), 2)
    qa["repair_rmse_tmax"] = round(float(np.sqrt(((fixed["tmax"] - fixed["tmax_true"]) ** 2).mean())), 2)
    clean.to_parquet(DATA / "processed" / "daily_clean.parquet", index=False)
    clean.to_csv(DATA / "processed" / "daily_clean.csv", index=False)
    json.dump(qa, open(DATA / "processed" / "quality_report.json", "w"), indent=2)
    print("   QA:", qa, flush=True)

    print("[3/5] simulating IoT AWS feed ...", flush=True)
    reg = station_registry()
    reg.to_csv(DATA / "processed" / "aws_stations.csv", index=False)
    aws = qc_aws(generate_aws(truth))
    aws.to_parquet(DATA / "processed" / "aws_hourly.parquet", index=False)
    print(f"   AWS rows: {len(aws):,}  qc flags: {aws['qc_flag'].value_counts().to_dict()}", flush=True)

    print("[4/5] climatological normals (train years only) ...", flush=True)
    normals = compute_normals(clean[clean["date"] <= TRAIN_END])

    print("[5/5] training models (Phase II) ...", flush=True)
    models.fit(clean, normals)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
