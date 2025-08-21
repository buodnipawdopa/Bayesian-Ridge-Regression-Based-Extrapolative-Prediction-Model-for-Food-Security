# -*- coding: utf-8 -*-
r"""
03_build_integrated_similarity.py
--------------------------------------------------
Combine 4×m coefficients (W) with 4-dim AHBCR vector (a) to obtain a single
m-dim exposure fingerprint per country:  f = W^T a;
then compute pairwise cosine similarity across countries.
Only computation and saving are performed (no plotting).

Inputs (relative to SUBMISSION_ROOT), per country:
  results/food2health_v2/<Country>/A_to_B_coeffs.csv  # supports long/wide (4×m or m×4)
  results/food2health_v2/<Country>/HBCR_metrics.csv   # uses AHBCR column for DALY, Death, Inc, Prev

Outputs (atomic replace):
  results/integrated/
    - integrated_exposure_fingerprint.csv        # rows=Country, cols=m indicators (f)
    - integrated_exposure_fingerprint_z.csv      # column-standardized F_z
    - aggregated_outcome_AHBCR.csv               # 4-domain AHBCR per country + Mean/SD/AbsSum
    - pairwise_similarity_cosine.csv             # country × country cosine similarity matrix
    - pairs_ranked.csv                           # unique pairs (i<j) with cosine similarity (desc)


"""

import os, glob, shutil, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def detect_root():
    env = os.environ.get("SUBMISSION_ROOT")
    cand = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "results")):
            return os.path.normpath(c)
    return r"D:\GBD MNBAC\Code"

SUBMISSION_ROOT = detect_root()
IN_ROOT  = os.path.join(SUBMISSION_ROOT, "results", "food2health_v2")
OUT_DIR  = os.path.join(SUBMISSION_ROOT, "results", "integrated")
TMP_DIR  = OUT_DIR + ".__tmp"

# Fresh TMP
os.makedirs(os.path.join(SUBMISSION_ROOT, "results"), exist_ok=True)
if os.path.isdir(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, exist_ok=True)

FOUR = ["DALY", "Death", "Inc", "Prev"]
ISO3 = {
    "Argentina":"ARG","Australia":"AUS","Austria":"AUT","Belgium":"BEL","Canada":"CAN",
    "Chile":"CHL","Cyprus":"CYP","Denmark":"DNK","Finland":"FIN","France":"FRA",
    "Germany":"DEU","Greece":"GRC","Iceland":"ISL","Ireland":"IRL","Israel":"ISR",
    "Italy":"ITA","Japan":"JPN","Luxembourg":"LUX","Malta":"MLT","Netherlands":"NLD",
    "New Zealand":"NZL","Norway":"NOR","Portugal":"PRT","Republic of Korea":"KOR",
    "Spain":"ESP","Sweden":"SWE","Switzerland":"CHE","United Kingdom":"GBR",
    "United States of America":"USA","Uruguay":"URY"
}

def read_W_4xM(fp: str) -> pd.DataFrame:
    """
    Read A_to_B_coeffs.csv and return a 4×m matrix W:
      rows = [DALY, Death, Inc, Prev]   (FOUR)
      cols = m exposure indicators
    Acceptable inputs:
      1) long format with columns {Domain, Indicator, Coef} -> pivot
      2) wide format (first column is row name). If wide includes FOUR as columns,
         rows are indicators -> transpose; otherwise rows are FOUR -> keep.
      3) fallback: try first column as index, handle similarly.
    """
    df = pd.read_csv(fp)
    if set(["Domain","Indicator","Coef"]).issubset(df.columns):
        W = df.pivot(index="Domain", columns="Indicator", values="Coef")
    elif df.columns[0].lower() in ("indicator","domain","unnamed: 0"):
        df = df.rename(columns={df.columns[0]:"Row"})
        if set(FOUR).issubset(df.columns):
            W = df.set_index("Row")[FOUR].T
        else:
            W = df.set_index("Row")
    else:
        df = pd.read_csv(fp, index_col=0)
        if set(FOUR).issubset(df.columns):
            W = df[FOUR].T
        else:
            W = df

    if not set(FOUR).issubset(set(map(str.strip, map(str, W.index)))):
        W.index = W.index.astype(str).str.strip()
    if not set(FOUR).issubset(W.index):
        raise ValueError(f"missing FOUR domains {FOUR} in: " + fp)
    W = W.loc[FOUR]
    return W  # 4×m

def read_ahbcr(fp: str) -> pd.Series:
    """Read HBCR_metrics.csv and return 4-dim AHBCR series in FOUR order."""
    df = pd.read_csv(fp, index_col=0)
    idx = [d for d in FOUR if d in df.index]
    if len(idx) != 4 or "AHBCR" not in df.columns:
        raise ValueError(f"missing FOUR AHBCR in: {fp}")
    a = df.loc[FOUR, "AHBCR"].astype(float)
    a.index = FOUR
    return a

def main():
    # Collect per-country fingerprints
    fingerprints = {}
    agg_outcomes = []

    # Expect per-country dirs under IN_ROOT
    if not os.path.isdir(IN_ROOT):
        raise SystemExit("Coefficient root not found: " + IN_ROOT)

    for fp_W in glob.glob(os.path.join(IN_ROOT, "*", "A_to_B_coeffs.csv")):
        country = os.path.basename(os.path.dirname(fp_W))
        fp_H = os.path.join(IN_ROOT, country, "HBCR_metrics.csv")
        if not os.path.isfile(fp_H):
            print("⚠ missing HBCR_metrics for", country, "-> skip")
            continue
        try:
            W = read_W_4xM(fp_W)    # 4×m
            a = read_ahbcr(fp_H)    # 4
            f = (W.T @ a.values).astype(float)   # m-dim fingerprint
            f.index = W.columns
            fingerprints[country] = f

            vals = a.values
            agg_outcomes.append({
                "Country": country, "DALY": a["DALY"], "Death": a["Death"],
                "Inc": a["Inc"], "Prev": a["Prev"],
                "Mean": float(np.mean(vals)),
                "SD": float(np.std(vals, ddof=0)),
                "AbsSum": float(np.sum(np.abs(vals))),
            })
        except Exception as e:
            print("⚠ skip", country, ":", e)

    if not fingerprints:
        raise SystemExit("No usable A_to_B_coeffs/HBCR_metrics found under: " + IN_ROOT)

    # Harmonize columns across countries; fill missing by column median
    all_cols = sorted(set().union(*[f.index for f in fingerprints.values()]))
    F = pd.DataFrame({c: fingerprints[c].reindex(all_cols) for c in fingerprints}).T
    F.columns.name = "Indicator"
    F = F.apply(lambda col: col.fillna(col.median()), axis=0)

    # Column-standardize to get Fz
    Fz = pd.DataFrame(StandardScaler().fit_transform(F), index=F.index, columns=F.columns)

    # Country × Country cosine similarity
    S = pd.DataFrame(cosine_similarity(Fz.values), index=F.index, columns=F.index)

    # Pairs (i<j), sorted desc by similarity
    pairs = []
    countries = list(S.index)
    for i in range(len(countries)):
        for j in range(i+1, len(countries)):
            ci, cj = countries[i], countries[j]
            pairs.append({"Country_i": ci, "Country_j": cj, "CosineSim": float(S.loc[ci, cj])})
    pairs_df = pd.DataFrame(pairs).sort_values("CosineSim", ascending=False)

    # Save to TMP (no plots)
    F.to_csv(os.path.join(TMP_DIR, "integrated_exposure_fingerprint.csv"), encoding="utf-8-sig")
    Fz.to_csv(os.path.join(TMP_DIR, "integrated_exposure_fingerprint_z.csv"), encoding="utf-8-sig")
    pd.DataFrame(agg_outcomes).to_csv(os.path.join(TMP_DIR, "aggregated_outcome_AHBCR.csv"),
                                      index=False, encoding="utf-8-sig")
    S.to_csv(os.path.join(TMP_DIR, "pairwise_similarity_cosine.csv"), encoding="utf-8-sig", float_format="%.6f")
    pairs_df.to_csv(os.path.join(TMP_DIR, "pairs_ranked.csv"), index=False, encoding="utf-8-sig", float_format="%.6f")

    # Atomic replace: delete old -> rename TMP -> final
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.rename(TMP_DIR, OUT_DIR)
    print("✅ Fingerprints & similarity saved ->", OUT_DIR)
    print("   Countries =", len(F), ", Indicators =", F.shape[1])

if __name__ == "__main__":
    main()
