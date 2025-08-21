# -*- coding: utf-8 -*-
r"""
00_exposure_lag_oneclick_minimal.py
--------------------------------------------------
One-click, minimal-output pipeline that COMBINES:
  (1) Validate exposure lag A_{t-k} → B_t for k = 0..5 (no lag of B),
  (2) Select a single global lag k_global based on MAE ratio vs lag=0,
and writes ONLY ONE file:
  results/exposure_lag_validation/selected_exposure_lag.json

Data layout (relative to the submission root):
  A: data/food_safety/<Region>/*_filtered.csv
  B: data/GBD_by_location_renamed/<Region>/
       {Prevalence, DALYs_(Disability-Adjusted_Life_Years), Deaths, Incidence}.csv

Defaults:
  - Train years: 2002–2016 (unified to 2006–2016 so k=5 is feasible)
  - Valid years: 2017–2021
  - Metric for selection: MAE (smaller is better)
  - Improvement threshold vs lag=0: 5% (if below, choose k_global=0)
  - Min support rate across (region, domain): 60%

"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ---------- Robust submission root detection (works with code runners) ----------
def detect_submission_root() -> str:
    """
    Try, in order:
      1) SUBMISSION_ROOT env,
      2) D:\\GBD MNBAC\\Code (Declared root),
      3) current working directory (if it already contains 'data').
    """
    env = os.environ.get("SUBMISSION_ROOT")
    candidates = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "data")):
            return os.path.normpath(c)
    # Fallback to declared root even if 'data' doesn't exist yet
    return r"D:\GBD MNBAC\Code"

# ---------- Config (defaults; args allow override but not required) ------------
DOM_MAP = {
    "Prevalence":                             "Prev",
    "DALYs_(Disability-Adjusted_Life_Years)": "DALY",
    "Deaths":                                 "Death",
    "Incidence":                              "Inc"
}
REG_KEYS  = {"region","Region","Area","location_name","Location"}
YEAR_KEYS = {"year","Year","Years","yr"}

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in REG_KEYS:  rename[c] = "region"
        if c in YEAR_KEYS: rename[c] = "year"
    return df.rename(columns=rename)

# ---------- Core pipeline (validate + select; in-memory; single JSON out) -------
def run_oneclick(metric: str = "MAE",
                 improv_thresh: float = 0.05,
                 min_support: float = 0.60) -> str:
    SUBMISSION_ROOT = detect_submission_root()
    ROOT_A  = os.path.join(SUBMISSION_ROOT, "data", "food_safety")
    ROOT_B  = os.path.join(SUBMISSION_ROOT, "data", "GBD_by_location_renamed")
    OUT_DIR = os.path.join(SUBMISSION_ROOT, "results", "exposure_lag_validation")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Sanity checks
    if not os.path.isdir(ROOT_A) or not os.path.isdir(ROOT_B):
        raise FileNotFoundError(
            "Input folders not found.\n"
            f"  ROOT_A: {ROOT_A}\n"
            f"  ROOT_B: {ROOT_B}\n"
            "Prepare data under SUBMISSION_ROOT/data before running."
        )

    LAGS = list(range(0, 6))
    TRAIN_YEARS_BASE = list(range(2002, 2017))
    VALID_YEARS      = list(range(2017, 2022))

    regions = [d for d in os.listdir(ROOT_A) if os.path.isdir(os.path.join(ROOT_A, d))]
    regions.sort()

    # Collect validation metrics per (region, lag, domain)
    records: List[Dict] = []

    for region in regions:
        # A: pick FIRST "*_filtered.csv" to mirror original logic
        a_files = glob.glob(os.path.join(ROOT_A, region, "*_filtered.csv"))
        if not a_files:
            continue
        A_df_raw = pd.read_csv(a_files[0])
        A_df = std_cols(A_df_raw).query("2001 <= year <= 2021").set_index("year")
        A_vars = [c for c in A_df.columns if c != "region"]

        # B: stack four domains 
        b_list = []
        for fname, tag in DOM_MAP.items():
            b_path = os.path.join(ROOT_B, region, f"{fname}.csv")
            if not os.path.exists(b_path):
                continue
            tmp = std_cols(pd.read_csv(b_path))[["year","val"]]
            tmp["domain"] = tag
            b_list.append(tmp)
        if not b_list:
            continue
        B_long = pd.concat(b_list, ignore_index=True)
        B_val  = B_long.pivot(index="year", columns="domain", values="val").sort_index()
        B_domains = B_val.columns.tolist()

        # Unify train window so that A_{t - max(LAGS)} exists
        min_year_A = int(A_df.index.min()) if len(A_df) else 2001
        max_lag    = max(LAGS)
        train_years = [y for y in TRAIN_YEARS_BASE if (y - max_lag) >= min_year_A]
        if not train_years:
            continue

        lag0_mae_by_domain = {}

        for lag in LAGS:
            rows_train, rows_valid = [], []

            # Train rows
            for yr in train_years:
                if (yr in B_val.index) and ((yr - lag) in A_df.index):
                    X_part = A_df.loc[yr - lag, A_vars]
                    y_part = B_val.loc[yr, B_domains]
                    row = pd.concat([pd.Series({"year": yr}), X_part, y_part])
                    rows_train.append(row)

            # Valid rows
            for yr in VALID_YEARS:
                if (yr in B_val.index) and ((yr - lag) in A_df.index):
                    X_part = A_df.loc[yr - lag, A_vars]
                    y_part = B_val.loc[yr, B_domains]
                    row = pd.concat([pd.Series({"year": yr}), X_part, y_part])
                    rows_valid.append(row)

            if (not rows_train) or (not rows_valid):
                continue

            train_df = pd.DataFrame(rows_train).set_index("year").sort_index()
            valid_df = pd.DataFrame(rows_valid).set_index("year").sort_index()

            X_train = train_df.drop(columns=B_domains)
            Y_train = train_df[B_domains]
            X_valid = valid_df.drop(columns=B_domains)
            Y_valid = valid_df[B_domains]

            # Standardize + train (MultiOutput BayesianRidge)
            xsc = StandardScaler().fit(X_train)
            ysc = StandardScaler().fit(Y_train)
            model = MultiOutputRegressor(BayesianRidge(tol=1e-5))
            model.fit(xsc.transform(X_train), ysc.transform(Y_train))

            # Predict valid set
            Y_pred_std = model.predict(xsc.transform(X_valid))
            Y_pred = ysc.inverse_transform(Y_pred_std)
            Y_pred = pd.DataFrame(Y_pred, index=Y_valid.index, columns=B_domains)

            # Per-domain metrics
            maes, rmses, r2s = {}, {}, {}
            for d in B_domains:
                maes[d]  = float(mean_absolute_error(Y_valid[d], Y_pred[d]))
                rmses[d] = float(np.sqrt(mean_squared_error(Y_valid[d], Y_pred[d])))
                r2s[d]   = float(r2_score(Y_valid[d], Y_pred[d]))
                records.append({
                    "region": region, "lag": lag, "domain": d,
                    "MAE": maes[d], "RMSE": rmses[d], "R2": r2s[d]
                })

            # cache baseline lag=0 MAE
            if lag == 0:
                lag0_mae_by_domain = maes.copy()

        # (Optional per-region lag existence flag was in original scripts; not needed for final JSON-only output.)

    # ---- Select global lag (in-memory) ----
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid records produced. Check data availability and year coverage.")

    # Merge with lag=0 baseline
    base = df[df["lag"] == 0][["region", "domain", metric]].rename(columns={metric: f"{metric}0"})
    dfm = df.merge(base, on=["region", "domain"], how="inner")
    if dfm.empty:
        raise RuntimeError("No comparable records (missing lag=0 baseline).")

    # ratio = metric_lag / metric_lag0  (smaller is better)
    dfm["ratio"] = dfm[metric] / dfm[f"{metric}0"]
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "lag"])

    # Aggregate scores per lag
    all_pairs = base.shape[0]
    rows = []
    for lag, sub in dfm.groupby("lag"):
        n_pairs = sub.shape[0]
        support = n_pairs / all_pairs if all_pairs > 0 else 0.0
        med     = float(sub["ratio"].median())
        p25     = float(sub["ratio"].quantile(0.25))
        p75     = float(sub["ratio"].quantile(0.75))
        mean    = float(sub["ratio"].mean())
        rows.append({
            "lag": int(lag),
            "n_pairs": int(n_pairs),
            "support_rate": support,
            "median_ratio": med,
            "p25": p25,
            "p75": p75,
            "mean_ratio": mean
        })
    score_df = pd.DataFrame(rows).sort_values(["median_ratio", "lag"]).reset_index(drop=True)

    # Apply min support & choose lag
    cand_df = score_df[score_df["support_rate"] >= min_support].copy()
    if cand_df.empty:
        cand_df = score_df.copy()
    best_row   = cand_df.iloc[cand_df["median_ratio"].argmin()]
    best_lag   = int(best_row["lag"])
    improvement = 1.0 - float(best_row["median_ratio"])  # improvement vs lag=0
    if improvement < improv_thresh:
        best_lag = 0

    # ---- Single JSON output with everything needed ----
    result = {
        "EXPOSURE_LAG": best_lag,
        "metric": metric,
        "improvement_vs_lag0_median": improvement,
        "improv_thresh": improv_thresh,
        "min_support": min_support,
        "n_regions_detected": int(df["region"].nunique()),
        "score_table": score_df.to_dict(orient="records")  # keep per-lag stats inside the single file
    }

    json_path = os.path.join(OUT_DIR, "selected_exposure_lag.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✓ Selected EXPOSURE_LAG =", best_lag)
    print("  Saved ->", json_path)
    return json_path

# ---------- CLI (optional; zero-arg works) ----------
def main():
    ap = argparse.ArgumentParser(description="One-click minimal pipeline: validate + select, single JSON output.")
    ap.add_argument("--metric", default="MAE", choices=["MAE","RMSE","R2"],
                    help="Metric for selection (MAE recommended).")
    ap.add_argument("--improv_thresh", type=float, default=0.05,
                    help="Improvement threshold vs lag=0 (if below, choose 0).")
    ap.add_argument("--min_support", type=float, default=0.60,
                    help="Minimum support rate across (region, domain).")
    args = ap.parse_args()
    run_oneclick(metric=args.metric, improv_thresh=args.improv_thresh, min_support=args.min_support)

if __name__ == "__main__":
    main()
