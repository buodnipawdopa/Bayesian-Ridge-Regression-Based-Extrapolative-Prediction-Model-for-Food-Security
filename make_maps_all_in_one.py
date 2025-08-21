# -*- coding: utf-8 -*-
r"""
make_maps_all_in_one.py — Hub / Links / Top-driver (map + legend only)
Changelog (per your request):
  • Hub: only sample countries are colored; non-sample countries stay as basemap.
          Within samples, tier=0 is PURPLE; tiers=1..4 are discrete colors (no gradient).
  • Links: ALL edges are ORANGE-RED; nodes keep 1–4 discrete tiers.
           Legend shows node tiers + one line sample for links (orange-red).
  • Top-driver: no grey in legend; use high-contrast colors only; color only countries with a driver.

I/O (relative to SUBMISSION_ROOT):
  inputs:  results/integrated/pairwise_similarity_cosine.csv
           results/integrated/integrated_exposure_fingerprint_z.csv
           data/ISO/region_abbreviations.xlsx   (optional mapping)
  outputs: visualizations/maps/
           hub_map.png, hub_legend.png,
           links_map.png, links_legend.png,
           top_driver_map.png, top_driver_legend.png
"""

import os, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import LineString

# prefer new colormap API; fallback gracefully
try:
    from matplotlib import colormaps as cm_new
except Exception:
    cm_new = None
import matplotlib.cm as cm_old

# ---------- Root detection ----------
def detect_submission_root():
    env = os.environ.get("SUBMISSION_ROOT")
    cand = [env, r"D:\GBD MNBAC\Code", os.getcwd()]
    for c in cand:
        if c and os.path.isdir(os.path.join(c, "results")):
            return os.path.normpath(c)
    return r"D:\GBD MNBAC\Code"

SUBMISSION_ROOT = detect_submission_root()
INTEG_DIR = os.path.join(SUBMISSION_ROOT, "results", "integrated")
VIZ_DIR   = os.path.join(SUBMISSION_ROOT, "visualizations", "maps")
os.makedirs(VIZ_DIR, exist_ok=True)

# ---------- ISO mapping ----------
FALLBACK_ISO = {
    "Argentina":"ARG","Australia":"AUS","Austria":"AUT","Belgium":"BEL","Canada":"CAN","Chile":"CHL",
    "Cyprus":"CYP","Denmark":"DNK","Finland":"FIN","France":"FRA","Germany":"DEU","Greece":"GRC",
    "Iceland":"ISL","Ireland":"IRL","Israel":"ISR","Italy":"ITA","Japan":"JPN","Luxembourg":"LUX",
    "Malta":"MLT","Netherlands":"NLD","New Zealand":"NZL","Norway":"NOR","Portugal":"PRT",
    "Republic of Korea":"KOR","Spain":"ESP","Sweden":"SWE","Switzerland":"CHE",
    "United Kingdom":"GBR","United States of America":"USA","Uruguay":"URY"
}
def load_iso_map():
    xls = os.path.join(SUBMISSION_ROOT, "data", "ISO", "region_abbreviations.xlsx")
    if not os.path.isfile(xls):
        return {}
    try:
        df = pd.read_excel(xls)
    except Exception:
        return {}
    cols = {str(c).strip().lower(): c for c in df.columns}
    name_keys = ("region","name","country","location","location_name")
    code_keys = ("iso","iso3","iso_3","abbr","code","short","iso_code")
    name_col = next((cols[k] for k in name_keys if k in cols), None)
    code_col = next((cols[k] for k in code_keys if k in cols), None)
    if not name_col or not code_col:
        return {}
    m = (df[[name_col, code_col]].dropna()
         .astype({name_col: "string", code_col: "string"}))
    m[name_col] = m[name_col].str.strip()
    m[code_col] = m[code_col].str.strip().str.upper()
    return dict(zip(m[name_col], m[code_col]))
ISO_MAP_EXCEL = load_iso_map()

def to_iso(name: str) -> str:
    if not isinstance(name, str):
        return None
    k = name.strip()
    if len(k) == 3 and k.isupper():
        return k
    if k in ISO_MAP_EXCEL:
        return ISO_MAP_EXCEL[k]
    if k in FALLBACK_ISO:
        return FALLBACK_ISO[k]
    kl = k.lower()
    for kk, vv in ISO_MAP_EXCEL.items():
        if str(kk).strip().lower() == kl:
            return vv
    for kk, vv in FALLBACK_ISO.items():
        if kk.lower() == kl:
            return vv
    return k[:3].upper()

# ---------- Basemap ----------
def read_world():
    """prefer geodatasets; fallback to Natural Earth (110m) with auto download."""
    try:
        import geodatasets as gds
        p = gds.get_path("naturalearth.countries")
        try:
            gdf = gpd.read_file(p, engine="pyogrio")
        except Exception:
            gdf = gpd.read_file(p)
    except Exception:
        cache = os.path.join(VIZ_DIR, "_cache_world")
        os.makedirs(cache, exist_ok=True)
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        zf  = os.path.join(cache, "ne_countries.zip")
        shp = os.path.join(cache, "ne_110m_admin_0_countries.shp")
        if not os.path.exists(shp):
            try:
                import urllib.request, zipfile
                if not os.path.exists(zf):
                    urllib.request.urlretrieve(url, zf)
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(cache)
            except Exception as e:
                raise RuntimeError("Basemap unavailable (need geodatasets or internet).") from e
        try:
            gdf = gpd.read_file(shp, engine="pyogrio")
        except Exception:
            gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(epsg=4326)
    for c in ["iso_a3","ADM0_A3","ISO_A3","adm0_a3","ADM0_A3_US"]:
        if c in gdf.columns:
            gdf["iso_a3"] = gdf[c].astype(str).str.upper()
            break
    if "iso_a3" not in gdf.columns:
        raise RuntimeError("Basemap missing ISO3 field.")
    gdf["rep_pt"] = gdf.representative_point()
    gdf["lon"] = gdf["rep_pt"].x
    gdf["lat"] = gdf["rep_pt"].y
    return gdf

def plot_base(ax, world_gdf):
    # basemap only (non-sample countries remain as this)
    world_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

# ---------- Palettes ----------
def get_discrete_cmap(name: str, n: int):
    if cm_new is not None:
        try:
            return cm_new.get_cmap(name).resampled(n)
        except Exception:
            pass
    return cm_old.get_cmap(name, n)

def discrete_palette_0_4():
    # 0 → 紫色（viridis 左端），1..4 依序更亮；用于 tiers 0..4
    cmap = get_discrete_cmap("viridis", 5)
    colors = cmap(np.linspace(0, 1, 5))
    return {i: tuple(colors[i]) for i in range(5)}

# —— Top‑driver 高对比色（去掉近似的第二个红色；仅保留 #d62728 一种“真红”）
BRIGHT_NOGREY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#00A6FF",
    "#4daf4a", "#984ea3", "#ff7f00", "#377eb8", "#a6cee3", "#f781bf"
]

# ---------- Standardize similarity ----------
def standardize_similarity(S_raw: pd.DataFrame, world: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = [to_iso(x) if isinstance(x, str) else x for x in S_raw.index]
    cols = [to_iso(x) if isinstance(x, str) else x for x in S_raw.columns]
    S = S_raw.copy()
    S.index, S.columns = rows, cols
    S = S.groupby(level=0).mean(numeric_only=True)
    S = S.T.groupby(level=0).mean(numeric_only=True).T
    iso_world = set(world["iso_a3"].astype(str))
    keep = sorted(list(set(S.index) & set(S.columns) & iso_world))
    S = S.loc[keep, keep].apply(pd.to_numeric, errors="coerce")
    S = (S + S.T) / 2.0
    np.fill_diagonal(S.values, 0.0)
    return S

# ---------- A) Hub map ----------
def draw_hub_map(S_iso: pd.DataFrame, th_hub: float, world: gpd.GeoDataFrame, dpi: int):
    # only compute on the S_iso set; others remain basemap (not colored)
    hub_count = (S_iso >= th_hub).sum(axis=1)
    hub_tier  = hub_count.clip(upper=4).astype(int)  # 0..4
    pal = discrete_palette_0_4()

    # highlight only sample countries
    highlight = world[world["iso_a3"].isin(hub_tier.index)].copy()
    highlight = highlight.merge(hub_tier.rename("HubTier"),
                                left_on="iso_a3", right_index=True, how="left")
    highlight["HubTier"] = highlight["HubTier"].astype(int)
    highlight["color"]   = highlight["HubTier"].map(pal)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    plot_base(ax, world)  # draw basemap
    highlight.plot(ax=ax, color=highlight["color"], edgecolor="white", linewidth=0.6)
    out_map = os.path.join(VIZ_DIR, "hub_map.png")
    plt.savefig(out_map, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # legend (0..4)
    fig, ax = plt.subplots(figsize=(4, 6), dpi=dpi); ax.axis("off")
    handles = [Patch(facecolor=pal[i], edgecolor="none", label=str(i)) for i in range(5)]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9, ncol=1,
              title=f"Neighbors (cos ≥ {th_hub:.2f})", title_fontsize=10)
    out_leg = os.path.join(VIZ_DIR, "hub_legend.png")
    plt.savefig(out_leg, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    return hub_tier, out_map, out_leg

# ---------- B) Links map ----------
def draw_links_map(S_iso: pd.DataFrame, world: gpd.GeoDataFrame, hi: float, dpi: int):
    # edges (≥ hi) from upper triangle
    idx = list(S_iso.index)
    edges = []
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            a, b = idx[i], idx[j]
            w = S_iso.iat[i, j]
            if pd.notna(w) and w >= hi:
                edges.append((a, b))
    # node degree (cap 4) & colors (tiers 1..4)
    deg = pd.Series(0, index=idx)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    deg_tier = deg.clip(upper=4).astype(int)

    pal = discrete_palette_0_4()
    ORANGE_RED = "#e64a19"  # all links use this color (as requested)

    sub = world[world["iso_a3"].isin(idx)].copy()
    coords = sub.set_index("iso_a3")[["lon","lat"]].to_dict("index")

    # build all lines once (single color)
    lines = []
    for a, b in edges:
        if a in coords and b in coords:
            pa = (coords[a]["lon"], coords[a]["lat"])
            pb = (coords[b]["lon"], coords[b]["lat"])
            lines.append(LineString([pa, pb]))

    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    plot_base(ax, world)
    if lines:
        gpd.GeoSeries(lines, crs=4326).plot(ax=ax, color=ORANGE_RED, linewidth=0.9, alpha=0.85)

    sub = sub.copy()
    sub["deg_tier"] = sub["iso_a3"].map(deg_tier).fillna(0).astype(int)
    sub = sub[sub["deg_tier"] > 0]  # only color nodes with degree>0
    if not sub.empty:
        sub["color"] = sub["deg_tier"].map(pal)
        sub.plot(ax=ax, color=sub["color"], edgecolor="white", linewidth=0.6)
    out_map = os.path.join(VIZ_DIR, "links_map.png")
    plt.savefig(out_map, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # legend: node tiers (1..4) + single orange-red link sample
    present = [i for i in [1,2,3,4] if (deg_tier == i).any()]
    fig, ax = plt.subplots(figsize=(5, 6), dpi=dpi); ax.axis("off")
    handles = [Patch(facecolor=pal[i], edgecolor="none", label=f"{i}") for i in present]
    handles += [Line2D([0],[0], color=ORANGE_RED, lw=2, label=f"links (cos ≥ {hi:.2f})")]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9, ncol=1,
              title="Node degree tiers", title_fontsize=10)
    out_leg = os.path.join(VIZ_DIR, "links_legend.png")
    plt.savefig(out_leg, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    return out_map, out_leg

# ---------- C) Top-driver map ----------
def draw_top_driver_map(fz: pd.DataFrame, world: gpd.GeoDataFrame, allowed_iso: list, dpi: int):
    # keep only countries present both in basemap and allowed_iso
    fz = fz.copy()
    fz.index = [to_iso(x) if isinstance(x, str) else x for x in fz.index]
    keep = sorted(set(fz.index) & set(world["iso_a3"]) & set(allowed_iso))
    fz = fz.loc[keep]
    if fz.empty:
        return None, None

    # top driver per country
    top = fz.abs().idxmax(axis=1).rename("TopDriver")
    g = world.merge(top, left_on="iso_a3", right_index=True, how="left")

    cats = sorted(g["TopDriver"].dropna().unique())
    # high-contrast palette (no grey) — now with only one true red
    colors = BRIGHT_NOGREY[:len(cats)]
    color_map = {c: colors[i] for i, c in enumerate(cats)}
    g["color"] = g["TopDriver"].map(color_map)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
    plot_base(ax, world)
    g_col = g[g["color"].notna()].copy()  # only color countries with a driver
    if not g_col.empty:
        g_col.plot(ax=ax, color=g_col["color"], edgecolor="white", linewidth=0.6)
    out_map = os.path.join(VIZ_DIR, "top_driver_map.png")
    plt.savefig(out_map, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # legend: categories only (no grey)
    fig, ax = plt.subplots(figsize=(5, 8), dpi=dpi); ax.axis("off")
    handles = [Patch(facecolor=color_map[c], edgecolor="none", label=c) for c in cats]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9, ncol=1,
              title="Top driver (|fingerprint_z| argmax)", title_fontsize=10)
    out_leg = os.path.join(VIZ_DIR, "top_driver_legend.png")
    plt.savefig(out_leg, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out_map, out_leg

# ---------- Orchestrator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th-hub", type=float, default=0.70, help="Hub neighbor threshold (cosine). Default=0.70")
    ap.add_argument("--hi",      type=float, default=None, help="Link threshold (cosine). Default=same as --th-hub")
    ap.add_argument("--dpi",     type=int,   default=300,  help="DPI for outputs. Default=300")
    args = ap.parse_args()
    HI = args.hi if args.hi is not None else args.th_hub

    S_path  = os.path.join(INTEG_DIR, "pairwise_similarity_cosine.csv")
    FZ_path = os.path.join(INTEG_DIR, "integrated_exposure_fingerprint_z.csv")
    if not os.path.isfile(S_path):
        raise FileNotFoundError("Missing: " + S_path)

    world = read_world()
    S_raw = pd.read_csv(S_path, index_col=0)
    S_iso = standardize_similarity(S_raw, world)
    allowed_iso = list(S_iso.index)

    # A) Hub
    hub_tier, hub_map, hub_leg = draw_hub_map(S_iso, args.th_hub, world, args.dpi)
    print("✓ Hub :", hub_map, "|", hub_leg)

    # B) Links
    links_map, links_leg = draw_links_map(S_iso, world, HI, args.dpi)
    print("✓ Links:", links_map, "|", links_leg)

    # C) Top driver
    if os.path.isfile(FZ_path):
        fz = pd.read_csv(FZ_path, index_col=0)
        td_map, td_leg = draw_top_driver_map(fz, world, allowed_iso, args.dpi)
        if td_map:
            print("✓ Top driver:", td_map, "|", td_leg)
        else:
            print("⚠ Top-driver map skipped (no matching rows).")
    else:
        print("⚠ Not found:", FZ_path, "— skip top-driver map.")

    print("\nALL MAPS SAVED ->", VIZ_DIR)

if __name__ == "__main__":
    main()
