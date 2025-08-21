# -*- coding: utf-8 -*-
r"""
01_run_food2health_v2.py
--------------------------------------------------
End-to-end main model (English & path-cleaned). It:
  1) Trains A→B mapping on 2001–2021 and exports coefficients per region
  2) End-to-end validation:
       - Fit linear trend on A in 2001–2016; extrapolate A to 2017–2021
       - Predict B for 2017–2021 via A→B model
       - Metrics: MAE / RMSE (per region), Cross-sectional R² per year (raw & calibrated)
  3) Extrapolates A to 2022–2040 using 2001–2021 trend
  4) Predicts B to 2022–2040; outputs UI 2.0, baseline, HBCR, coefficients
Result layout (relative to SUBMISSION_ROOT):
  - data/food_safety/<Region>/*_filtered.csv
  - data/GBD_by_location_renamed/<Region>/{Prevalence,DALYs_(Disability-Adjusted_Life_Years),Deaths,Incidence}.csv
  - results/food2health_v2/                         (per-region outputs)
  - results/food2health_v2_validation/              (global validation outputs)
Atomic replace:
  - All new outputs are written into temp folders first.
  - Only after success do we DELETE old result folders and RENAME temp -> final.
"""

import os, glob, shutil, json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------- robust root detection --------
def detect_root():
    env = os.environ.get("SUBMISSION_ROOT")
    cand = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "data")):
            return os.path.normpath(c)
    return r"D:\GBD MNBAC\Code"

SUBMISSION_ROOT = detect_root()
ROOT_A  = os.path.join(SUBMISSION_ROOT, "data", "food_safety")
ROOT_B  = os.path.join(SUBMISSION_ROOT, "data", "GBD_by_location_renamed")

RES_BASE    = os.path.join(SUBMISSION_ROOT, "results")
OUT_FINAL   = os.path.join(RES_BASE, "food2health_v2")
VAL_FINAL   = os.path.join(RES_BASE, "food2health_v2_validation")
OUT_TMP     = OUT_FINAL + ".__tmp"
VAL_TMP     = VAL_FINAL + ".__tmp"

os.makedirs(RES_BASE, exist_ok=True)
os.makedirs(OUT_TMP, exist_ok=True)
os.makedirs(VAL_TMP, exist_ok=True)

# -------- hyperparams & helpers --------
RUN_MODEL = True
YEARS_FUT = np.arange(2022, 2041)

TRAIN_START, TRAIN_END = 2001, 2016
VAL_START,   VAL_END   = 2017, 2021
DO_VALIDATION = True

REG_KEYS  = {"region", "Region", "Area", "location_name", "Location"}
YEAR_KEYS = {"year", "Year", "Years", "yr"}

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in REG_KEYS:  rename[c] = "region"
        if c in YEAR_KEYS: rename[c] = "year"
    return df.rename(columns=rename)

DOM_MAP = {
    "Prevalence":                             "Prev",
    "DALYs_(Disability-Adjusted_Life_Years)": "DALY",
    "Deaths":                                 "Death",
    "Incidence":                              "Inc"
}
B_TAGS = ["Prev","DALY","Death","Inc"]

def mae_rmse_matrix(Y_true_df: pd.DataFrame, Y_pred_df: pd.DataFrame):
    idx = Y_true_df.index.intersection(Y_pred_df.index)
    cols = [c for c in Y_true_df.columns if c in Y_pred_df.columns]
    if len(idx)==0 or len(cols)==0: return np.nan, np.nan
    T = Y_true_df.loc[idx, cols].astype(float).values
    P = Y_pred_df.loc[idx, cols].astype(float).values
    m = np.isfinite(T) & np.isfinite(P)
    if m.sum()==0: return np.nan, np.nan
    Tf = T[m]; Pf = P[m]
    mae  = float(mean_absolute_error(Tf, Pf))
    rmse = float(np.sqrt(mean_squared_error(Tf, Pf)))
    return mae, rmse

def cross_sectional_r2_by_year(pool_valid_df: pd.DataFrame,
                               pool_train_df: pd.DataFrame,
                               do_calibration: bool = True):
    from sklearn.linear_model import LinearRegression
    rows = []
    calib = {}
    if do_calibration and not pool_train_df.empty:
        for d, sub in pool_train_df.groupby("domain"):
            sub = sub.dropna(subset=["y_true","y_pred_raw"])
            if len(sub) >= 5:
                lr = LinearRegression().fit(sub[["y_pred_raw"]].values, sub["y_true"].values)
                calib[d] = (float(lr.intercept_), float(lr.coef_[0]))
            else:
                calib[d] = (0.0, 1.0)
    else:
        calib = {d:(0.0,1.0) for d in pool_valid_df["domain"].unique()}

    years = sorted(pool_valid_df["year"].unique().tolist())
    for d in sorted(pool_valid_df["domain"].unique().tolist(), key=lambda x: B_TAGS.index(x) if x in B_TAGS else 0):
        a, b = calib.get(d, (0.0,1.0))
        for yr in years:
            sub = pool_valid_df[(pool_valid_df["domain"]==d) & (pool_valid_df["year"]==yr)] \
                               .dropna(subset=["y_true","y_pred_raw"])
            if len(sub) >= 5:
                y  = sub["y_true"].values
                yh = sub["y_pred_raw"].values
                yh_cal = a + b * yh
                def _r2(u, v):
                    if len(u) < 2 or np.var(u)==0: return np.nan
                    return float(1 - np.sum((u-v)**2)/np.sum((u-np.mean(u))**2))
                r2_raw = _r2(y, yh); r2_cal = _r2(y, yh_cal)
            else:
                r2_raw = r2_cal = np.nan
            rows.append({"domain": d, "year": int(yr),
                         "R2_cs_raw": r2_raw, "R2_cs_cal": r2_cal})

    by_year_df = pd.DataFrame(rows)
    sums = []
    for d, sub in by_year_df.groupby("domain"):
        sums.append({
            "domain": d,
            "R2_cs_raw_median": float(sub["R2_cs_raw"].median(skipna=True)),
            "R2_cs_cal_median": float(sub["R2_cs_cal"].median(skipna=True)),
            "R2_cs_raw_mean":   float(sub["R2_cs_raw"].mean(skipna=True)),
            "R2_cs_cal_mean":   float(sub["R2_cs_cal"].mean(skipna=True)),
        })
    sum_df = pd.DataFrame(sums)
    if not sum_df.empty:
        overall = {
            "domain": "OVERALL_median4",
            "R2_cs_raw_median": float(sum_df["R2_cs_raw_median"].median(skipna=True)),
            "R2_cs_cal_median": float(sum_df["R2_cs_cal_median"].median(skipna=True)),
            "R2_cs_raw_mean":   float(sum_df["R2_cs_raw_mean"].median(skipna=True)),
            "R2_cs_cal_mean":   float(sum_df["R2_cs_cal_mean"].median(skipna=True)),
        }
        sum_df = pd.concat([sum_df, pd.DataFrame([overall])], ignore_index=True)
    return by_year_df, sum_df

def replace_dir(final_dir: str, tmp_dir: str):
    """Delete final_dir (if exists) and rename tmp_dir -> final_dir."""
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.rename(tmp_dir, final_dir)

def main():
    # Collect regions from A
    regions = [d for d in os.listdir(ROOT_A) if os.path.isdir(os.path.join(ROOT_A, d))]

    POOL_TRAIN, POOL_VALID = [], []
    for region in regions:
        print(f"\n→ Processing region: {region}")
        out_dir = os.path.join(OUT_TMP, region)
        os.makedirs(out_dir, exist_ok=True)

        # A
        a_path = glob.glob(os.path.join(ROOT_A, region, "*_filtered.csv"))
        if not a_path:
            print("  [WARN] No A file. Skip.")
            continue
        A_df   = std_cols(pd.read_csv(a_path[0])).query("2001 <= year <= 2021")
        A_vars = [c for c in A_df.columns if c not in ("region", "year")]

        # B
        b_list, b_files = [], []
        for raw, tag in DOM_MAP.items():
            fp = os.path.join(ROOT_B, region, f"{raw}.csv")
            if not os.path.exists(fp):
                continue
            tmp = std_cols(pd.read_csv(fp))[["year","val","upper","lower"]]
            tmp["domain"] = tag
            b_list.append(tmp); b_files.append(fp)
        if not b_list:
            print("  [WARN] No B files. Skip.")
            continue

        # docs summary (kept for audit)
        def doc_info(path: str, df: pd.DataFrame) -> dict:
            return {
                "file": path, "rows": len(df), "cols": len(df.columns),
                "columns": list(df.columns), "missing_cells": int(df.isna().sum().sum())
            }
        docs_summary = [doc_info(a_path[0], A_df)] + [doc_info(fp, std_cols(pd.read_csv(fp))) for fp in b_files]
        pd.DataFrame(docs_summary).to_csv(os.path.join(out_dir, "docs_summary.csv"), index=False, encoding="utf-8-sig")

        if not RUN_MODEL:
            print("  ✓ Saved docs summary (skip modeling).")
            continue

        # ---- B long→wide, 2001–2021
        B_long = pd.concat(b_list, ignore_index=True).query("2001 <= year <= 2021")
        B_val  = B_long.pivot(index="year", columns="domain", values="val")
        B_lo   = B_long.pivot(index="year", columns="domain", values="lower")
        B_up   = B_long.pivot(index="year", columns="domain", values="upper")
        B_domains = [c for c in B_val.columns if c in B_TAGS]
        B_val = B_val[B_domains]; B_lo = B_lo[B_domains]; B_up = B_up[B_domains]

        # ---- Full-history 2001–2021 model per region
        rows_full = []
        for yr in range(2001, 2022):
            if yr not in A_df["year"].values or yr not in B_val.index:
                continue
            X_part = A_df.set_index("year").loc[yr, A_vars]
            y_part = B_val.loc[yr]
            rows_full.append(pd.concat([pd.Series({"year": yr}), X_part, y_part]))
        if not rows_full:
            print("  [WARN] No overlapping years. Skip.")
            continue
        XY_full = pd.DataFrame(rows_full).set_index("year")
        X_full, Y_full = XY_full[A_vars], XY_full[B_domains]

        xsc = StandardScaler().fit(X_full)
        ysc = StandardScaler().fit(Y_full)
        model = MultiOutputRegressor(BayesianRidge(tol=1e-5))
        model.fit(xsc.transform(X_full), ysc.transform(Y_full))

        # export coefficients
        coef_mat = pd.DataFrame(index=A_vars, columns=B_domains, dtype=float)
        for i, dom in enumerate(B_domains):
            coef_mat[dom] = model.estimators_[i].coef_[: len(A_vars)]
        coef_mat.to_csv(os.path.join(out_dir, "A_to_B_coeffs.csv"), float_format="%.6f", encoding="utf-8-sig", index_label="Indicator")

        # persist model & scalers (optional but useful)
        import joblib
        joblib.dump(model, os.path.join(out_dir, "model.pkl"))
        joblib.dump(xsc,  os.path.join(out_dir, "X_scaler.pkl"))
        joblib.dump(ysc,  os.path.join(out_dir, "Y_scaler.pkl"))

        # ---- End-to-end validation
        if DO_VALIDATION:
            X_tr = A_df.set_index("year").loc[TRAIN_START:TRAIN_END, A_vars]
            Y_tr = B_val.loc[TRAIN_START:TRAIN_END, B_domains]

            xsc_v = StandardScaler().fit(X_tr)
            ysc_v = StandardScaler().fit(Y_tr)
            mdl_v = MultiOutputRegressor(BayesianRidge(tol=1e-5))
            mdl_v.fit(xsc_v.transform(X_tr), ysc_v.transform(Y_tr))

            # (a) train window: observed A -> B (for calibration)
            Yp_tr_std = mdl_v.predict(xsc_v.transform(X_tr))
            Yp_tr = pd.DataFrame(ysc_v.inverse_transform(Yp_tr_std), index=Y_tr.index, columns=B_domains)
            for d in B_domains:
                for yr in Y_tr.index:
                    yt = float(Y_tr.loc[yr, d]); yp = float(Yp_tr.loc[yr, d])
                    if np.isfinite(yt) and np.isfinite(yp):
                        POOL_TRAIN.append({"region": region, "domain": d, "year": int(yr), "y_true": yt, "y_pred_raw": yp})

            # (b) extrapolate A trend to validation years
            years_tr = np.arange(TRAIN_START, TRAIN_END + 1).reshape(-1, 1)
            years_va = np.arange(VAL_START,   VAL_END   + 1).reshape(-1, 1)
            A_hat_va = pd.DataFrame(index=np.arange(VAL_START, VAL_END + 1), columns=A_vars, dtype=float)
            for v in A_vars:
                yv = X_tr[v].values
                if np.isfinite(yv).sum() >= 3:
                    lr = LinearRegression().fit(years_tr, yv)
                    A_hat_va[v] = lr.predict(years_va)
                else:
                    A_hat_va[v] = np.nan

            # (c) validation prediction
            Y_va = B_val.loc[VAL_START:VAL_END, B_domains]
            Yp_va_std = mdl_v.predict(xsc_v.transform(A_hat_va))
            Yp_va = pd.DataFrame(ysc_v.inverse_transform(Yp_va_std), index=Y_va.index, columns=B_domains)

            mae_e2e, rmse_e2e = mae_rmse_matrix(Y_va, Yp_va)
            row_region = {"region": region, "MAE_e2e": mae_e2e, "RMSE_e2e": rmse_e2e}
            # global validation table (append)
            gval = os.path.join(VAL_TMP, "e2e_mae_rmse_all_regions.csv")
            pd.DataFrame([row_region]).to_csv(gval, index=False, mode="a",
                                              header=not os.path.exists(gval), encoding="utf-8-sig")
            # per-region
            pd.DataFrame([row_region]).to_csv(os.path.join(out_dir, "validation_mae_rmse.csv"), index=False, encoding="utf-8-sig")

            # pool for cross-sectional R²
            for d in B_domains:
                for yr in Y_va.index:
                    yt = float(Y_va.loc[yr, d]); yp = float(Yp_va.loc[yr, d])
                    if np.isfinite(yt) and np.isfinite(yp):
                        POOL_VALID.append({"region": region, "domain": d, "year": int(yr), "y_true": yt, "y_pred_raw": yp})

        # ---- A trend extrapolation: 2022–2040
        A_future = pd.DataFrame({
            v: LinearRegression().fit(A_df[["year"]], A_df[v]).predict(YEARS_FUT.reshape(-1, 1))
            for v in A_vars
        }, index=YEARS_FUT)

        # ---- Predict B to 2022–2040 (mean & sigma)
        preds, sigs = [], []
        for yr in YEARS_FUT:
            X_now = A_future.loc[yr, A_vars].values.reshape(1, -1)
            y_std, s_std = [], []
            for est in model.estimators_:
                p, s = est.predict(xsc.transform(X_now), return_std=True)
                y_std.append(p[0]); s_std.append(s[0])
            y_pred = ysc.inverse_transform(np.array(y_std).reshape(1, -1)).flatten()
            sigma  = np.array(s_std) * ysc.scale_
            preds.append(y_pred); sigs.append(sigma)

        pred_df  = pd.DataFrame(preds, index=YEARS_FUT, columns=B_domains)
        sigma_df = pd.DataFrame(sigs, index=YEARS_FUT, columns=[f"sigma_{d}" for d in B_domains])

        # ---- UI 2.0
        ui_rows = []
        for d in B_domains:
            w21 = max(B_up[d].loc[2021] - B_val[d].loc[2021], B_val[d].loc[2021] - B_lo[d].loc[2021])
            w22, w_inf = w21 * 1.05, np.sqrt(w21**2 + sigma_df[f"sigma_{d}"].iloc[0]**2)
            for k, yr in enumerate(YEARS_FUT, 1):
                w = w22 if k == 1 else w22 + (k - 1) / 18 * (w_inf - w22)
                mu = pred_df.loc[yr, d]
                ui_rows.append({"year": yr, "domain": d, "val": mu, "lower": mu - w, "upper": mu + w})
        UI_df = pd.DataFrame(ui_rows)

        # ---- Baseline (linear fit on B 2001–2021)
        baseline_df = pd.DataFrame({
            d: LinearRegression().fit(np.arange(2001, 2022).reshape(-1, 1), B_val[d].loc[2001:2021]).predict(YEARS_FUT.reshape(-1, 1))
            for d in B_domains
        }, index=YEARS_FUT)

        # ---- HBCR metrics
        hbcr = {}
        for d in B_domains:
            hist21, pred40, base40 = B_val[d].loc[2021], pred_df[d].loc[2040], baseline_df[d].loc[2040]
            HBCR   = (pred40 - hist21) / hist21
            BHBCR  = (base40 - hist21) / hist21
            AHBCR  = HBCR - BHBCR
            hbcr[d] = {"HBCR": HBCR, "BHBCR": BHBCR, "TC": "Yes" if HBCR * BHBCR > 0 else "No", "AHBCR": AHBCR}
        hbcr_df = pd.DataFrame(hbcr).T

        # ---- Save per-region outputs to TMP
        pred_df.to_csv(os.path.join(out_dir, "pred_2022-2040.csv"), encoding="utf-8-sig")
        UI_df.to_csv(os.path.join(out_dir, "UI_2022-2040.csv"), index=False, encoding="utf-8-sig")
        baseline_df.to_csv(os.path.join(out_dir, "baseline_2022-2040.csv"), encoding="utf-8-sig")
        hbcr_df.to_csv(os.path.join(out_dir, "HBCR_metrics.csv"), encoding="utf-8-sig")
        print("  ✓ Saved region outputs:", out_dir)

    # ---- Global cross-sectional R² (write to TMP validation dir) ----
    POOL_TRAIN_DF = pd.DataFrame(POOL_TRAIN)
    POOL_VALID_DF = pd.DataFrame(POOL_VALID)
    if not POOL_VALID_DF.empty:
        by_year_df, sum_df = cross_sectional_r2_by_year(POOL_VALID_DF, POOL_TRAIN_DF, do_calibration=True)
        by_year_df.to_csv(os.path.join(VAL_TMP, "cross_sectional_R2_by_year.csv"), index=False, encoding="utf-8-sig")
        sum_df.to_csv(os.path.join(VAL_TMP, "cross_sectional_R2_summary.csv"), index=False, encoding="utf-8-sig")

    # ---- Atomic replace: delete old -> rename TMP -> FINAL ----
    # Only replace if TMP contains something meaningful (at least one region dir or a validation file)
    has_region = any(os.path.isdir(os.path.join(OUT_TMP, d)) for d in os.listdir(OUT_TMP)) if os.path.isdir(OUT_TMP) else False
    has_val = any(os.path.isfile(os.path.join(VAL_TMP, f)) for f in os.listdir(VAL_TMP)) if os.path.isdir(VAL_TMP) else False
    if not has_region and not has_val:
        raise RuntimeError("No new outputs produced; abort replacing old results.")

    # Remove old and swap
    if os.path.isdir(OUT_FINAL):
        import shutil; shutil.rmtree(OUT_FINAL)
    if os.path.isdir(VAL_FINAL):
        import shutil; shutil.rmtree(VAL_FINAL)
    os.rename(OUT_TMP, OUT_FINAL)
    os.rename(VAL_TMP, VAL_FINAL)
    print("\n✓ Replaced old results with NEW results.")
    print("  Regions ->", OUT_FINAL)
    print("  Global  ->", VAL_FINAL)

if __name__ == "__main__":
    main()
