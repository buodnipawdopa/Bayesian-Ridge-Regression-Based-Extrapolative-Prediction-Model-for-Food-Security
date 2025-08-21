# -*- coding: utf-8 -*-
"""
make_visuals_oneclick.py — 4-in-1 visuals (integrated similarity: heatmap only)
-------------------------------------------------------------------------------
A) Per‑region trends (2001–2040) with continuous baseline
B) Domain mosaics (All_<Domain>_2001-2040.png)
C) Region classification scatter with INTELLIGENT ORANGE
   (orange if the four AHBCR values include both positive and negative;
    otherwise blue if mean<=0, red if mean>0)
D) Importance posterior visuals (if available)
   - posterior_mean_hdi.png  (posterior mean with 95% HDI) + numeric labels
   - prob_positive.png       (P(Coef>0), 4-band coloring with lines at 0.05/0.50/0.95)
                              + per-indicator numeric labels
E) Integrated similarity: heatmap only
   - similarity_heatmap.png  (cosine matrix; optional hierarchical ordering)
     * X‑axis tick labels rotated 45°, shown as ISO codes
     * Tick font size configurable via HEATMAP_TICK_FONTSIZE
     * Cell values annotated to two decimals
"""

import os, glob, shutil, warnings, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil, sqrt

# Optional libs for similarity heatmap
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except Exception:
    linkage = leaves_list = None

# ======= Global style options =======
HEATMAP_TICK_FONTSIZE = int(os.environ.get("HEATMAP_TICK_FONTSIZE", "9"))

# ---------- Robust root detection ----------
def detect_submission_root():
    env = os.environ.get("SUBMISSION_ROOT")
    candidates = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "results")):
            return os.path.normpath(c)
    return r"D:\GBD MNBAC\Code"

SUBMISSION_ROOT = detect_submission_root()

# ---------- Paths ----------
DATA_B   = os.path.join(SUBMISSION_ROOT, "data", "GBD_by_location_renamed")
MODEL_R  = os.path.join(SUBMISSION_ROOT, "results", "food2health_v2")
IMP_R    = os.path.join(SUBMISSION_ROOT, "results", "importance")
INTEG_R  = os.path.join(SUBMISSION_ROOT, "results", "integrated")
VIZ_ROOT = os.path.join(SUBMISSION_ROOT, "visualizations")
VIZ_REG  = os.path.join(VIZ_ROOT, "regions")
VIZ_PAN  = os.path.join(VIZ_ROOT, "panels")
VIZ_DASH = os.path.join(VIZ_ROOT, "dashboard")
VIZ_IMP  = os.path.join(VIZ_ROOT, "importance")
VIZ_SIM  = os.path.join(VIZ_ROOT, "integrated_similarity")
for d in (VIZ_REG, VIZ_PAN, VIZ_DASH, VIZ_IMP, VIZ_SIM):
    os.makedirs(d, exist_ok=True)

# Clean previous visualizations (only inside visualizations/)
def clean_folder(folder):
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            except Exception:
                pass

for _d in (VIZ_REG, VIZ_PAN, VIZ_DASH, VIZ_IMP, VIZ_SIM):
    clean_folder(_d)

# ---------- Domain map & colors ----------
DOM_MAP = {
    "Prevalence": "Prev",
    "DALYs_(Disability-Adjusted_Life_Years)": "DALY",
    "Deaths": "Death",
    "Incidence": "Inc"
}
DOM_ORDER = ["Inc", "Prev", "DALY", "Death"]
COLORS = {"Prev":"#1f77b4","DALY":"#ff7f0e","Death":"#2ca02c","Inc":"#d62728"}

# ---------- Column standardization helpers ----------
REG_KEYS  = {"region","Region","Area","location_name","Location"}
YEAR_KEYS = {"year","Year","Years","yr"}
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in REG_KEYS:  rename[c] = "region"
        if c in YEAR_KEYS: rename[c] = "year"
    return df.rename(columns=rename)

# ---------- ISO mapping (Excel preferred at data/ISO/region_abbreviations.xlsx) ----------
def load_iso_map():
    """
    Preferred path: <SUBMISSION_ROOT>/data/ISO/region_abbreviations.xlsx
    Fallbacks: <SUBMISSION_ROOT>/region_abbreviations.xlsx, <SUBMISSION_ROOT>/metadata/region_abbreviations.xlsx
    Name column (case-insensitive): Region / Name / Country / Location / Location_Name
    Code column (case-insensitive): ISO / ISO3 / ISO_3 / Abbr / Code / Short / Iso_Code
    """
    candidates = [
        os.path.join(SUBMISSION_ROOT, "data", "ISO", "region_abbreviations.xlsx"),
        os.path.join(SUBMISSION_ROOT, "region_abbreviations.xlsx"),
        os.path.join(SUBMISSION_ROOT, "metadata", "region_abbreviations.xlsx"),
    ]
    name_keys = ("region","name","country","location","location_name")
    code_keys = ("iso","iso3","iso_3","abbr","code","short","iso_code")

    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            df = pd.read_excel(p)
        except Exception:
            continue
        cols_lc = {str(c).strip().lower(): c for c in df.columns}
        name_col = next((cols_lc[k] for k in name_keys if k in cols_lc), None)
        code_col = next((cols_lc[k] for k in code_keys if k in cols_lc), None)
        if not name_col or not code_col:
            continue
        m = (df[[name_col, code_col]]
             .dropna()
             .astype({name_col: "string", code_col: "string"}))
        m[name_col] = m[name_col].str.strip()
        m[code_col] = m[code_col].str.strip().str.upper()
        return dict(zip(m[name_col], m[code_col])), p
    return {}, None

def map_iso_series(names: pd.Index) -> pd.Series:
    iso_map, _ = load_iso_map()
    def map_one(name: str) -> str:
        k = str(name).strip()
        if k in iso_map:
            return iso_map[k]
        kl = k.lower()
        for kk, vv in iso_map.items():
            if str(kk).strip().lower() == kl:
                return vv
        return k[:3].upper()  # fallback
    return pd.Series([map_one(n) for n in names], index=names, name="ISO")

# ---------- A) Per-region trend plots ----------
def render_region_trends():
    regions = [d for d in os.listdir(MODEL_R) if os.path.isdir(os.path.join(MODEL_R, d))]
    if not regions:
        print("WARNING: No region folders found under:", MODEL_R)
        return

    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    for region in regions:
        r_dir = os.path.join(MODEL_R, region)
        ui_path    = os.path.join(r_dir, "UI_2022-2040.csv")
        base_path  = os.path.join(r_dir, "baseline_2022-2040.csv")
        if not os.path.exists(ui_path):
            continue

        # 1) UI predictions 2022–2040
        ui_df = pd.read_csv(ui_path)
        ui_df = std_cols(ui_df)[["year","domain","val","lower","upper"]]

        # 2) Historical B 2001–2021
        hist_list = []
        for fname, tag in DOM_MAP.items():
            f = os.path.join(DATA_B, region, f"{fname}.csv")
            if not os.path.exists(f): continue
            df = std_cols(pd.read_csv(f))
            df["domain"] = tag
            hist_list.append(df[["year","domain","val","lower","upper"]])
        if not hist_list:
            continue
        hist_df = pd.concat(hist_list, ignore_index=True)
        hist_df = hist_df[(hist_df.year>=2001)&(hist_df.year<=2021)].sort_values("year")

        # 3) Linear baseline 2001–2040 (historical segment = actual values)
        full_base = None
        if os.path.exists(base_path):
            base_raw = pd.read_csv(base_path, index_col=0)
            base_rows = []
            for dom in base_raw.columns:
                for yr, val in base_raw[dom].items():
                    base_rows.append({"year": int(yr), "domain": dom, "baseline": val})
            base_df = pd.DataFrame(base_rows)
            hist_base = hist_df[["year","domain","val"]].rename(columns={"val":"baseline"})
            full_base = pd.concat([hist_base, base_df], ignore_index=True).sort_values("year")

        comb_df = pd.concat([hist_df, ui_df], ignore_index=True).sort_values("year")

        # 4) Plot per domain & save under VIZ_REG/<Region>/
        out_dir = os.path.join(VIZ_REG, region)
        os.makedirs(out_dir, exist_ok=True)

        for dom in comb_df["domain"].unique():
            df_dom = comb_df[comb_df.domain==dom]
            color  = COLORS.get(dom, "gray")

            fig, ax = plt.subplots(figsize=(8,4))
            ax.fill_between(df_dom.year, df_dom.lower, df_dom.upper, color=color, alpha=0.15)
            ax.plot(df_dom.year, df_dom.val, "-", color=color, linewidth=2)

            if full_base is not None:
                fb = full_base[full_base.domain==dom]
                ax.plot(fb.year, fb.baseline, "--", color=color, linewidth=2)

            ax.axvline(2021, color="black", linestyle=":", linewidth=1)
            ax.set_xlim(2001, 2040); ax.set_xticks([2001, 2021, 2040])
            ax.set_title(region)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

            out_png = os.path.join(out_dir, f"{region}_{dom}_2001-2040.png")
            plt.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)
        print("✓ Region trends ->", out_dir)

# ---------- B) Domain mosaics ----------
def stitch_domain_panels():
    regions = [d for d in os.listdir(VIZ_REG) if os.path.isdir(os.path.join(VIZ_REG, d))]
    if not regions:
        print("WARNING: No per-region PNGs found under:", VIZ_REG)
        return

    TEMPLATE = "{region}_{dom}_2001-2040.png"
    DPI = 300
    for dom in ["Inc","Prev","DALY","Death"]:
        img_paths = []
        for reg in regions:
            p = os.path.join(VIZ_REG, reg, TEMPLATE.format(region=reg, dom=dom))
            if os.path.isfile(p): img_paths.append(p)
        if not img_paths:
            print(f"WARNING: No PNGs for domain {dom}; skip panel."); continue

        n = len(img_paths); cols = ceil(sqrt(n)); rows = ceil(n / cols)
        first_img = mpimg.imread(img_paths[0]); h_px, w_px = first_img.shape[:2]
        fig_w = (w_px * cols) / DPI; fig_h = (h_px * rows) / DPI

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=DPI,
                                 gridspec_kw={'wspace': 0, 'hspace': 0})
        axes = np.array(axes).reshape(-1)
        for ax in axes[len(img_paths):]: ax.remove()
        for ax, path in zip(axes, img_paths):
            ax.imshow(mpimg.imread(path)); ax.axis('off')

        out_png = os.path.join(VIZ_PAN, f"All_{dom}_2001-2040.png")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(out_png, dpi=DPI); plt.close(fig)
        print("✓ Panel saved:", out_png)

# ---------- C) Region classification scatter with intelligent ORANGE ----------
def build_region_scatter():
    # Prefer integrated summary; else derive from HBCR_metrics.csv
    agg_path = os.path.join(SUBMISSION_ROOT, "results", "integrated", "aggregated_outcome_AHBCR.csv")
    if os.path.exists(agg_path):
        df = pd.read_csv(agg_path)
        if "Country" in df.columns: df = df.set_index("Country")
    else:
        rows = []
        regions = [d for d in os.listdir(MODEL_R) if os.path.isdir(os.path.join(MODEL_R, d))]
        for region in regions:
            hbcr_path = os.path.join(MODEL_R, region, "HBCR_metrics.csv")
            if not os.path.isfile(hbcr_path): continue
            try:
                t = pd.read_csv(hbcr_path, index_col=0)
                vals = {dom: float(t.loc[dom, "AHBCR"]) for dom in ["DALY","Death","Inc","Prev"] if dom in t.index}
                if len(vals) != 4: continue
                v = np.array([vals["DALY"], vals["Death"], vals["Inc"], vals["Prev"]], dtype=float)
                rows.append({
                    "Region": region,
                    "DALY": v[0], "Death": v[1], "Inc": v[2], "Prev": v[3],
                    "Mean": float(np.mean(v)), "SD": float(np.std(v, ddof=0)),
                    "AbsSum": float(np.sum(np.abs(v)))
                })
            except Exception:
                continue
        if not rows:
            print("WARNING: No HBCR summaries available; skip scatter."); return
        df = pd.DataFrame(rows).set_index("Region")

    # ISO mapping: Excel preferred, fallback to first 3 letters
    df["ISO"] = map_iso_series(df.index)

    # ORANGE rule: four domain values include both positive and negative
    def mixed_sign(row):
        vals = [row.get(k, np.nan) for k in ["DALY","Death","Inc","Prev"]]
        vals = [x for x in vals if pd.notna(x)]
        return (len(vals) == 4) and (min(vals) < 0) and (max(vals) > 0)
    df["MixSign"] = df.apply(mixed_sign, axis=1)
    df["ColorTag"] = np.where(df["MixSign"], "orange",
                        np.where(df["Mean"] <= 0, "blue", "red"))

    # sizes
    sizes = 300.0 * (df["AbsSum"] / df["AbsSum"].max())
    blue_face = (0.45, 0.65, 1.00, 0.20); blue_edge = (0.45, 0.65, 1.00, 0.40)
    red_face  = (1.00, 0.54, 0.54, 0.20); red_edge  = (1.00, 0.54, 0.54, 0.40)
    orange_face = (1.00, 0.60, 0.10, 0.25); orange_edge = (1.00, 0.60, 0.10, 0.60)

    left   = df[(df["ColorTag"]=="blue")]
    right  = df[(df["ColorTag"]=="red")]
    orange = df[(df["ColorTag"]=="orange")]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(left["Mean"],  left["SD"],  s=300.0 * left["AbsSum"]/df["AbsSum"].max(),
               facecolors=blue_face, edgecolors=blue_edge, linewidth=0.8, zorder=2)
    ax.scatter(right["Mean"], right["SD"], s=300.0 * right["AbsSum"]/df["AbsSum"].max(),
               facecolors=red_face,  edgecolors=red_edge,  linewidth=0.8, zorder=2)
    ax.scatter(orange["Mean"], orange["SD"], s=300.0 * orange["AbsSum"]/df["AbsSum"].max(),
               facecolors=orange_face, edgecolors=orange_edge, linewidth=0.8, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    # labels
    try:
        from adjustText import adjust_text
        texts = []
        for subset in (left, right, orange):
            for _, r in subset.iterrows():
                texts.append(ax.text(r["Mean"], r["SD"], r["ISO"], fontsize=8, zorder=4))
        adjust_text(texts, ax=ax, only_move={'points': 'y', 'text': 'xy'},
                    arrowprops=dict(arrowstyle="-", lw=0.5, color='grey'))
    except Exception:
        for i, (_, r) in enumerate(df.iterrows()):
            dy = 0.001 * (-1 if i % 2 else 1)
            ax.text(r["Mean"], r["SD"] + dy, r["ISO"], fontsize=8, zorder=4)

    ax.set_xlabel("Burden‑Weighted Mean AHBCR")
    ax.set_ylabel("Burden‑Weighted SD")
    plt.tight_layout()
    png_out = os.path.join(VIZ_DASH, "AHBCR_scatter_smart.png")
    fig.savefig(png_out, dpi=300); plt.close(fig)

    # classification table
    out_csv = os.path.join(VIZ_DASH, "region_classification_summary.csv")
    df_out = df[["ISO","DALY","Death","Inc","Prev","Mean","SD","AbsSum","MixSign","ColorTag"]].copy()
    df_out.to_csv(out_csv, encoding="utf-8-sig")
    print("✓ Scatter & summary ->", VIZ_DASH)

# ---------- D) Importance posterior visuals ----------
def render_importance_posteriors():
    """
    Read:
      results/importance/posterior_indicator.csv
      results/importance/prob_positive.csv
    Save PNGs to: visualizations/importance/
    """
    post_path = os.path.join(IMP_R, "posterior_indicator.csv")
    ppos_path = os.path.join(IMP_R, "prob_positive.csv")
    if not (os.path.isfile(post_path) and os.path.isfile(ppos_path)):
        print("WARNING: Importance posterior files not found under:", IMP_R)
        return

    def load_and_clean(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Indicator" in df.columns:
            df = df.set_index("Indicator")
        else:
            df = df.set_index(df.columns[0])
        df.index = (df.index.astype(str)
                          .str.replace(r"^Indicator\[(.*)\]$", r"\1", regex=True)
                          .str.strip())
        return df

    post = load_and_clean(post_path)
    ppos = load_and_clean(ppos_path)

    common = post.index.intersection(ppos.index)
    if len(common) == 0:
        print("WARNING: No common indicators between posterior and probability tables.")
        return
    post = post.loc[common]
    ppos = ppos.loc[common]

    need_post = {"mean","hdi_2.5%","hdi_97.5%"}
    need_ppos = {"P_Pos"}
    if not need_post.issubset(post.columns) or not need_ppos.issubset(ppos.columns):
        print("WARNING: Missing expected columns in posterior/probability files; skip plotting.")
        return

    # Figure 1: posterior mean with 95% HDI + numeric labels per row
    post["sign_flag"] = (post["mean"] >= 0).astype(int)  # 0 negative, 1 non-negative
    post_sorted = post.sort_values(["sign_flag", "mean"], ascending=[True, True])
    neg = post_sorted[post_sorted["sign_flag"] == 0]
    pos = post_sorted[post_sorted["sign_flag"] == 1]

    fig, ax = plt.subplots(figsize=(16, 8))
    if len(neg):
        ax.errorbar(neg["mean"], neg.index,
                    xerr=[neg["mean"] - neg["hdi_2.5%"], neg["hdi_97.5%"] - neg["mean"]],
                    fmt="o", capsize=4, color="#2E86C1", ecolor="#2E86C1")
    if len(pos):
        ax.errorbar(pos["mean"], pos.index,
                    xerr=[pos["mean"] - pos["hdi_2.5%"], pos["hdi_97.5%"] - pos["mean"]],
                    fmt="o", capsize=4, color="#C0392B", ecolor="#C0392B")

    # zero line
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)

    # annotate numeric mean[lo,hi] for each indicator
    lo_min = float(post_sorted["hdi_2.5%"].min())
    hi_max = float(post_sorted["hdi_97.5%"].max())
    span   = float(hi_max - lo_min) if np.isfinite(hi_max - lo_min) else 1.0
    margin = 0.02 * span
    for name, row in post_sorted.iterrows():
        m = float(row["mean"]); lo = float(row["hdi_2.5%"]); hi = float(row["hdi_97.5%"])
        s = f"{m:.2f}[{lo:.2f},{hi:.2f}]"
        x = max(m, lo, hi) + margin
        ax.text(x, name, s, va="center", ha="left", fontsize=10)

    ax.set_xlabel("Posterior Mean (Coef)")
    # expand xlim to fit text comfortably
    ax.set_xlim(lo_min - 0.05*span, hi_max + 0.35*span)
    plt.tight_layout()
    out1 = os.path.join(VIZ_IMP, "posterior_mean_hdi.png")
    plt.savefig(out1, dpi=300); plt.close(fig)

    # Figure 2: P(Coef>0) barh — numeric labels on bars
    ppos_sorted = ppos.sort_values("P_Pos", ascending=True)

    BLUE_STRONG = (0.00, 0.45, 0.75, 0.50)  # p < 0.05
    BLUE_WEAK   = (0.00, 0.45, 0.75, 0.25)  # 0.05 ≤ p < 0.50
    RED_WEAK    = (0.85, 0.10, 0.10, 0.25)  # 0.50 ≤ p < 0.95
    RED_STRONG  = (0.85, 0.10, 0.10, 0.50)  # p ≥ 0.95

    def classify_color(p):
        if p < 0.05:
            return "Strong negative (p<0.05)", BLUE_STRONG
        elif p < 0.50:
            return "Leaning negative (0.05–0.50)", BLUE_WEAK
        elif p < 0.95:
            return "Leaning positive (0.50–0.95)", RED_WEAK
        else:
            return "Strong positive (p≥0.95)", RED_STRONG

    labels_colors = ppos_sorted["P_Pos"].apply(classify_color)
    ppos_sorted["class"] = labels_colors.apply(lambda x: x[0])
    ppos_sorted["color"] = labels_colors.apply(lambda x: x[1])
    colors = ppos_sorted["color"].tolist()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(ppos_sorted.index, ppos_sorted["P_Pos"], color=colors, edgecolor='none')
    ax.invert_yaxis()
    ax.set_xlabel("Posterior Probability (Coef > 0)")

    # guide lines
    ax.axvline(0.05, linestyle="--", linewidth=2, color="grey")
    ax.axvline(0.50, linestyle="--", linewidth=1.5, color="grey")
    ax.axvline(0.95, linestyle="--", linewidth=2, color="grey")

    # numeric labels for each indicator
    ax.set_xlim(0, 1.05)
    for name, p in ppos_sorted["P_Pos"].items():
        p = float(p)
        if p > 0.96:
            x = min(1.0, p) - 0.02; ha = "right"
        else:
            x = p + 0.02; ha = "left"
        ax.text(x, name, f"{p:.2f}", va="center", ha=ha, fontsize=10)

    # legend: only classes present
    from matplotlib.patches import Patch
    legend_order = [
        ("Strong negative (p<0.05)", BLUE_STRONG),
        ("Leaning negative (0.05–0.50)", BLUE_WEAK),
        ("Leaning positive (0.50–0.95)", RED_WEAK),
        ("Strong positive (p≥0.95)", RED_STRONG),
    ]
    present = list(dict.fromkeys(ppos_sorted["class"]))
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for label, color in legend_order if label in present]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    out2 = os.path.join(VIZ_IMP, "prob_positive.png")
    plt.savefig(out2, dpi=300); plt.close(fig)

    print("✓ Importance figures ->", VIZ_IMP)
    print("  -", os.path.basename(out1))
    print("  -", os.path.basename(out2))

# ---------- E) Integrated similarity heatmap (lower-triangle with in-cell numbers) ----------
def render_integrated_similarity():
    """
    Read pairwise similarity matrix and draw a lower‑triangle heatmap with
    in‑cell numeric annotations (two decimals). The upper triangle is masked.
    Tick‑label size is controlled by HEATMAP_TICK_FONTSIZE (env var).
    Per‑cell size can be tuned with HEATMAP_CELL_INCH (env var, default 0.35).
    """
    S_path = os.path.join(INTEG_R, "pairwise_similarity_cosine.csv")
    if not os.path.isfile(S_path):
        print("WARNING: pairwise_similarity_cosine.csv not found under:", INTEG_R)
        return

    # load + clean
    S = pd.read_csv(S_path, index_col=0).apply(pd.to_numeric, errors="coerce")
    # fill diagonal if missing and drop empty rows/cols
    for i in S.index:
        if pd.isna(S.loc[i, i]): S.loc[i, i] = 1.0
    keep_rows = S.index[S.notna().sum(axis=1) > 0]
    S = S.loc[keep_rows, S.columns.intersection(keep_rows)]
    if S.empty:
        print("WARNING: similarity matrix is empty after cleaning; skip.")
        return

    # optional hierarchical ordering (average linkage on 1 - cosine)
    try:
        if linkage is not None:
            D = 1 - S.values
            Z = linkage(D, method="average")
            order = leaves_list(Z)
            S_ord = S.iloc[order, :].iloc[:, order]
        else:
            raise RuntimeError
    except Exception:
        S_ord = S.copy()

    # Map to ISO codes
    iso_rows = map_iso_series(S_ord.index)
    iso_cols = map_iso_series(S_ord.columns)
    S_ord.index = iso_rows.values
    S_ord.columns = iso_cols.values

    # Keep only the lower triangle by masking the upper triangle (k=1 excludes diagonal)
    mask = np.triu(np.ones_like(S_ord, dtype=bool), k=1)

    # ---- figure size & annotation font size heuristics ----
    n = S_ord.shape[0]
    cell_inch = float(os.environ.get("HEATMAP_CELL_INCH", "0.50"))  # per‑cell size in inches
    fig_w = max(6.0, cell_inch * n)
    fig_h = max(5.0, cell_inch * n)
    # annotation font size: shrink as n grows, but keep within [6, 12]
    annot_fs = int(max(6, min(12, 22 - 0.4 * n)))
    tick_fs = HEATMAP_TICK_FONTSIZE

    if sns is not None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        hm = sns.heatmap(
            S_ord, vmin=-1, vmax=1, cmap="coolwarm", square=True,
            mask=mask, annot=True, fmt=".2f",
            annot_kws={"size": annot_fs, "ha": "center", "va": "center"},
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.85}, xticklabels=True, yticklabels=True, ax=ax
        )
        # rotate & size ticks
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45); lbl.set_ha('right'); lbl.set_fontsize(tick_fs)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(tick_fs)
        # clip annotation texts to the axes (prevents overdraw outside cells/axes)
        for txt in ax.texts:
            txt.set_clip_on(True)

        ax.set_title("Country-to-Country Similarity (cosine)")
        plt.tight_layout()

    else:
        # Matplotlib fallback (manual annotation on a masked array)
        data = np.ma.array(S_ord.values, mask=mask)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(data, vmin=-1, vmax=1, cmap="coolwarm")
        # grid lines
        ax.set_xticks(np.arange(n+1)-0.5, minor=True)
        ax.set_yticks(np.arange(n+1)-0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        # ticks/labels
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(S_ord.columns, rotation=45, ha='right', fontsize=tick_fs)
        ax.set_yticklabels(S_ord.index, fontsize=tick_fs)
        ax.set_title("Country-to-Country Similarity (cosine)")
        # annotate only lower triangle (i >= j)
        for i in range(n):
            for j in range(n):
                if j <= i and not pd.isna(S_ord.iat[i, j]):
                    ax.text(j, i, f"{S_ord.iat[i, j]:.2f}",
                            ha="center", va="center", fontsize=annot_fs, clip_on=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

    out_heat = os.path.join(VIZ_SIM, "similarity_heatmap.png")
    plt.savefig(out_heat, dpi=300)
    plt.close()
    print("✓ Heatmap (lower-triangle, with numbers) ->", out_heat)

# ---------- Orchestrator ----------
def main():
    print("[Paths]")
    print("  SUBMISSION_ROOT :", SUBMISSION_ROOT)
    print("  DATA_B          :", DATA_B)
    print("  MODEL_R         :", MODEL_R)
    print("  IMP_R           :", IMP_R)
    print("  INTEG_R         :", INTEG_R)
    print("  VIZ_ROOT        :", VIZ_ROOT)
    print("  ISO Excel       :", os.path.join(SUBMISSION_ROOT, 'data', 'ISO', 'region_abbreviations.xlsx'))
    print("  HEATMAP_TICK_FONTSIZE:", HEATMAP_TICK_FONTSIZE)

    print("\nA) Rendering per-region trend plots ...")
    render_region_trends()

    print("\nB) Stitching domain panels ...")
    stitch_domain_panels()

    print("\nC) Building region classification scatter ...")
    build_region_scatter()

    print("\nD) Rendering importance posterior visuals ...")
    render_importance_posteriors()

    print("\nE) Rendering integrated similarity heatmap ...")
    render_integrated_similarity()

    print("\nALL VISUALIZATIONS DONE")
    print("Outputs ->", VIZ_ROOT)

if __name__ == "__main__":
    main()
