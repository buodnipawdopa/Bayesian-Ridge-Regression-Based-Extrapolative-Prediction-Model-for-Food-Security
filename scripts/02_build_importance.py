# -*- coding: utf-8 -*-
r"""
02_build_importance.py
--------------------------------------------------
Build "importance/consistency" tables from per-region coefficients.

Inputs (relative to SUBMISSION_ROOT):
  results/food2health_v2/<Region>/A_to_B_coeffs.csv

Outputs (atomic replace):
  results/importance/
    - consistency_table.csv
    - (optional if bambi/arviz installed)
        posterior_indicator.csv
        prob_positive.csv

Changes in this version:
  • Bayesian model uses NO-INTERCEPT coding so all 11 indicators appear:
      Coef ~ 0 + Indicator + (0 + Indicator | Country) + (1 | Domain)
  • Reindex posterior outputs to the full list of 11 indicator levels,
    ensuring 11 rows in both CSVs.
"""

import os, warnings, shutil, numpy as np, pandas as pd

def detect_root():
    env = os.environ.get("SUBMISSION_ROOT")
    cand = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "results")):
            return os.path.normpath(c)
    return r"D:\GBD MNBAC\Code"

SUBMISSION_ROOT = detect_root()
COEF_ROOT = os.path.join(SUBMISSION_ROOT, "results", "food2health_v2")
OUT_FINAL = os.path.join(SUBMISSION_ROOT, "results", "importance")
OUT_TMP   = OUT_FINAL + ".__tmp"

os.makedirs(os.path.join(SUBMISSION_ROOT, "results"), exist_ok=True)
# fresh TMP
if os.path.isdir(OUT_TMP):
    shutil.rmtree(OUT_TMP)
os.makedirs(OUT_TMP, exist_ok=True)

DOMAIN_MAP = {
    "DALY":"DALY", "Death":"Death",
    "Inc":"Incidence", "Incidence":"Incidence",
    "Prev":"Prevalence", "Prevalence":"Prevalence"
}

def collect_data():
    records = []
    if not os.path.isdir(COEF_ROOT):
        raise RuntimeError("Coefficient root not found: " + COEF_ROOT)
    for region in os.listdir(COEF_ROOT):
        fp = os.path.join(COEF_ROOT, region, "A_to_B_coeffs.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0":"Indicator"})
        keep = [c for c in df.columns if c in DOMAIN_MAP]
        if len(keep) < 4 or "Indicator" not in df.columns:
            warnings.warn(f"{fp} schema mismatch; skipped"); continue
        long = (df[["Indicator"]+keep].melt("Indicator", keep, var_name="Domain", value_name="Coef"))
        long["Domain"]  = long["Domain"].map(DOMAIN_MAP)
        long["Country"] = region
        records.append(long[["Country","Domain","Indicator","Coef"]])
        print("✓ read", fp)
    if not records:
        raise RuntimeError("No usable coefficient files found under: " + COEF_ROOT)
    out = pd.concat(records, ignore_index=True)
    # Standardize: ensure string type and strip whitespaces
    out["Indicator"] = out["Indicator"].astype(str).str.strip()
    out["Domain"]    = out["Domain"].astype(str).str.strip()
    out["Country"]   = out["Country"].astype(str).str.strip()
    return out

def consistency(df: pd.DataFrame):
    df = df.copy()
    # Avoid division by zero when variance = 0
    def safe_std(x): 
        s = x.std(ddof=0)
        return s if np.isfinite(s) and s > 0 else np.nan
    z = df.groupby(["Country","Domain"])["Coef"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else np.inf))
    df["t"] = z

    out = (df.groupby(["Indicator","Domain"])
             .agg(N_Pos=("Coef", lambda x: (x > 0).sum()),
                  N_Neg=("Coef", lambda x: (x < 0).sum()),
                  N_t2 =("t",    lambda x: (np.abs(x) > 2).sum()),
                  Total=("Coef","count"),
                  Mean =("Coef","mean"))
             .reset_index())
    out["P_SignMajor"] = out[["N_Pos","N_Neg"]].max(axis=1) / out["Total"].replace(0, np.nan)
    out["P_t_over2"]   = out["N_t2"] / out["Total"].replace(0, np.nan)
    out["Direction"]   = np.where(out["N_Pos"] > out["N_Neg"], "Positive", "Negative")
    out["Tag"]         = np.where((out["P_SignMajor"] >= 0.7) & (out["P_t_over2"] >= 0.7),
                                  "Robust " + out["Direction"], "Inconclusive")
    out.to_csv(os.path.join(OUT_TMP, "consistency_table.csv"), index=False, encoding="utf-8-sig")
    print("✓ consistency_table.csv saved")

def bayes(df: pd.DataFrame):
    try:
        import bambi as bmb, arviz as az
    except ImportError:
        print("⚠ bambi/arviz not installed; skipping Bayesian section")
        return

     # Ensure categorical levels are fixed (for complete 11 indicators output)
    df = df.copy()
    df["Indicator"] = df["Indicator"].astype("category")
     # Capture full indicator levels for reindexing
    indicator_levels = list(df["Indicator"].cat.categories)

    for c in ["Country","Domain"]:
        df[c] = df[c].astype("category")

    # —— Key change: no-intercept coding so all indicators are estimated —— #
    model = bmb.Model(
        "Coef ~ 0 + Indicator + (0 + Indicator | Country) + (1 | Domain)",
        df, family="gaussian",
        priors={"Indicator": bmb.Prior("Normal", mu=0, sigma=1)}
    )
    trace = model.fit(draws=1500, tune=1000, chains=4, cores=4, target_accept=0.9)

    # 1) Summary (reindexed to complete 11 indicators)
    summ = az.summary(trace, var_names=["Indicator"], hdi_prob=0.95)
     # az.summary index looks like "Indicator[xxx]", reindex to all levels
    full_idx = [f"Indicator[{lvl}]" for lvl in indicator_levels]
    summ = summ.reindex(full_idx)
    summ.to_csv(os.path.join(OUT_TMP, "posterior_indicator.csv"), encoding="utf-8-sig")

    # 2) P_Pos: from xarray to DataFrame (index = indicator levels)
    ppos = (trace.posterior["Indicator"] > 0).mean(dim=("chain","draw")).to_dataframe(name="P_Pos")
    ppos = ppos.reindex(indicator_levels)
    ppos.to_csv(os.path.join(OUT_TMP, "prob_positive.csv"), encoding="utf-8-sig")
    print("✓ Bayesian outputs saved (11 indicators)")

def main():
    data = collect_data()
    consistency(data)
    bayes(data)

    # ---- Atomic replace: delete old -> rename TMP -> FINAL ----
    if os.path.isdir(OUT_FINAL):
        shutil.rmtree(OUT_FINAL)
    os.rename(OUT_TMP, OUT_FINAL)
    print("\n🎉 Finished. Results:", OUT_FINAL)

if __name__ == "__main__":
    main()
