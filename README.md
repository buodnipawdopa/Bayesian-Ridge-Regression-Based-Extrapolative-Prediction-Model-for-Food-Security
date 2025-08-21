# Food–Health-MNBAC (High‑Income Countries): End‑to‑End Pipeline

**Scope.** This repository implements an end‑to‑end, fully reproducible pipeline that links macro food‑security indicators to four health outcomes (DALYs, deaths, incidence, prevalence) for high‑income countries, performs forecasting to 2040, and produces a compact set of figures and tables suitable for manuscript use.

**Key outputs.**

* Per‑country historical and forecast trends (with a history‑only baseline).
* Importance/consistency of exposure–outcome coefficients (frequentist + Bayesian views).
* Country “exposure fingerprints” and country–country similarity (cosine).
* Three global maps (hub tiers, high‑agreement links, top drivers).

---

## 1) Repository layout

```
.
├─ data/
│  ├─ GBD_by_location_renamed/<Region>/
│  │  ├─ Prevalence.csv
│  │  ├─ DALYs_(Disability-Adjusted_Life_Years).csv
│  │  ├─ Deaths.csv
│  │  └─ Incidence.csv
│  └─ ISO/region_abbreviations.xlsx     # flexible column names (Region/ISO, etc.)
├─ results/
│  ├─ food2health_v2/<Region>/
│  │  ├─ A_to_B_coeffs.csv
│  │  ├─ UI_2022-2040.csv
│  │  ├─ baseline_2022-2040.csv
│  │  └─ HBCR_metrics.csv
│  ├─ importance/
│  │  ├─ consistency_table.csv
│  │  ├─ posterior_indicator.csv
│  │  └─ prob_positive.csv
│  └─ integrated/
│     ├─ integrated_exposure_fingerprint.csv
│     ├─ integrated_exposure_fingerprint_z.csv
│     ├─ aggregated_outcome_AHBCR.csv
│     ├─ pairwise_similarity_cosine.csv
│     └─ pairs_ranked.csv
├─ visualizations/
│  ├─ regions/<Region>/*_2001-2040.png
│  ├─ panels/All_<Domain>_2001-2040.png
│  ├─ dashboard/
│  │  ├─ AHBCR_scatter_smart.png
│  │  └─ region_classification_summary.csv
│  ├─ importance/
│  │  ├─ posterior_mean_hdi.png
│  │  └─ prob_positive.png
│  ├─ integrated_similarity/similarity_heatmap.png
│  └─ maps/
│     ├─ hub_map.png, hub_legend.png
│     ├─ links_map.png, links_legend.png
│     └─ top_driver_map.png, top_driver_legend.png
├─ 00_exposure_lag_oneclick_minimal.py
├─ 01_run_food2health_v2.py
├─ 02_build_importance.py
├─ 03_build_integrated_similarity.py
├─ make_visuals_oneclick.py
└─ make_maps_all_in_one.py
```

> The two visualization drivers implement, respectively, the 4‑in‑1 figure bundle (including annotated posterior plots and a lower‑triangle, annotated similarity heatmap) and the three map products (hub/links/top‑driver). &#x20;

---

## 2) Data prerequisites

* **Outcomes (GBD 2021)**: DALYs, deaths, incidence, prevalence — country‑year time series (2001–2021) stored under `data/GBD_by_location_renamed/<Region>/`.
* **Exposures (FAOSTAT Food Security)**: 11 curated indicators in original units, aligned to 2001–2021 and harmonized with GBD by year.
* **ISO codes**: Optional mapping in `data/ISO/region_abbreviations.xlsx`. Column names are flexible; the scripts auto‑detect reasonable “Region/ISO” pairs. &#x20;

> **Note.** The pipeline uses rates for all outcomes and “all ages” health data to eliminate cross‑country population‑size artifacts. All data have been screened and preprocessed..(per manuscript methods).

---

## 3) Environment & dependencies

* **Python**: 3.10+ recommended.
* **Core**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* **Optional / recommended**:

  * Visuals: `seaborn`, `scipy` (for hierarchical ordering in the heatmap), `adjustText`
  * Maps: `geopandas`, `shapely`, `geodatasets` (or Natural Earth fallback), and either `pyogrio` or `fiona` for I/O
  * Bayesian importance (optional): `bambi`, `arviz`

Install (typical):

```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy adjustText
pip install geopandas shapely geodatasets pyogrio   # or: pip install fiona (instead of pyogrio)
# Optional for Bayesian aggregation in '02_build_importance.py':
pip install bambi arviz
```

---

## 4) Configuration

The scripts auto‑detect the project root, but you can **explicitly** set it:

```bash
# macOS / Linux
export SUBMISSION_ROOT="$(pwd)"

# Windows (PowerShell)
$env:SUBMISSION_ROOT = (Get-Location).Path
```

* **Heatmap font size**: `HEATMAP_TICK_FONTSIZE` (default `9`).
* **Heatmap cell size** (lower‑triangle view): `HEATMAP_CELL_INCH` (default `0.50`). Larger values yield larger cells and more legible in‑cell numbers.&#x20;

---

## 5) Reproducible run order

1. **(Optional) Global lag selection**

   ```bash
   python 00_exposure_lag_oneclick_minimal.py
   ```

   Produces a minimal JSON documenting the selected exposure–outcome lag (the analysis fixes a single global lag for the entire pipeline).

2. **Main model per country**

   ```bash
   python 01_run_food2health_v2.py
   ```

   Writes per‑country coefficients, forecasts, and the history‑only baseline under `results/food2health_v2/<Region>/`.

3. **Coefficient importance & consistency**

   ```bash
   python 02_build_importance.py
   ```

   Generates `results/importance/consistency_table.csv` and, if Bayesian packages are present, posterior summaries `posterior_indicator.csv` and `prob_positive.csv`.

4. **Integrated fingerprint & similarity**

   ```bash
   python 03_build_integrated_similarity.py
   ```

   Constructs country‑level exposure fingerprints (11‑D) and cosine similarity (`pairwise_similarity_cosine.csv`) plus handy long‑form pairs.

5. **One‑click figure bundle**

   ```bash
   python make_visuals_oneclick.py
   ```

   * Trends (per region/domain), domain mosaics, AHBCR scatter (with intelligent orange), and **importance** figures:

     * *Posterior mean with 95% HDI* now prints **`mean [lo, hi]`** per indicator on the figure.&#x20;
     * *P(Coef>0)* barh uses **four probability bands** with vertical guides at **0.05 / 0.50 / 0.95** and prints the **numeric probability** on each bar.&#x20;
   * **Integrated similarity heatmap** is rendered as a **lower‑triangle** matrix with **in‑cell numeric annotations (two decimals)**; cell and font size are tunable via env vars.&#x20;

6. **Maps (hub / links / top‑driver)**

   ```bash
   python make_maps_all_in_one.py --th-hub 0.70 --dpi 300
   ```

   * **Hub**: only sample countries colored; tier **0–4** shown as discrete classes (no gradient).
   * **Links**: **all edges in orange‑red**; node tiers (1–4) as discrete classes in the legend.
   * **Top‑driver**: high‑contrast palette, **no grey**; countries colored only where a driver exists.
     Outputs land in `visualizations/maps/`.&#x20;

---

## 6) What’s new in this version

* **Posterior mean (HDI) figure** now annotates each row with `mean [lo, hi]`.&#x20;
* **P(Coef>0)** bar chart prints the probability value on each bar, with four evidence bands and guide lines at 0.05 / 0.50 / 0.95.&#x20;
* **Similarity heatmap** switches to a **lower‑triangle** view with **two‑decimal in‑cell numbers**, and exposes sizing via `HEATMAP_CELL_INCH` and `HEATMAP_TICK_FONTSIZE`.&#x20;
* **Maps**: hub tiers as discrete classes (including tier 0), links drawn uniformly in **orange‑red**, and a high‑contrast top‑driver legend without grey.&#x20;

---

## 7) Outputs & interpretation at a glance

* **`visualizations/regions/`**: Domain‑specific trends per country (2001–2040), with history‑only baseline overlay and the 2021 anchor.
* **`visualizations/panels/`**: Grids of all countries per domain.
* **`visualizations/dashboard/AHBCR_scatter_smart.png`**: Mean vs SD of within‑country AHBCR (bubble ∝ absolute magnitude), with “intelligent orange” for mixed signs.
* **`visualizations/importance/`**:

  * `posterior_mean_hdi.png`: shows posterior mean and 95% HDI per indicator **and** the numeric `mean [lo, hi]`.&#x20;
  * `prob_positive.png`: four‑band evidence view with **numbers on bars** and guides at 0.05/0.50/0.95.&#x20;
* **`visualizations/integrated_similarity/similarity_heatmap.png`**: Lower‑triangle cosine similarity with **two‑decimal** in‑cell values; optional hierarchical ordering; ISO labels; tunable sizes.&#x20;
* **`visualizations/maps/`**:

  * `hub_map.png`, `hub_legend.png`: neighbor counts above a cosine threshold mapped to tiers 0–4 (discrete).&#x20;
  * `links_map.png`, `links_legend.png`: high‑agreement links (cos ≥ threshold) in **orange‑red**; node tiers 1–4.&#x20;
  * `top_driver_map.png`, `top_driver_legend.png`: top‑driver per country using a bright, non‑grey palette.&#x20;

---

## 8) Troubleshooting

* **Missing basemap**: `make_maps_all_in_one.py` prefers `geodatasets`; if absent, it tries to auto‑download Natural Earth 110m. Install `geodatasets` or ensure internet access on first run.&#x20;
* **ISO labels look off**: check `data/ISO/region_abbreviations.xlsx` (column names are flexible); the scripts also fall back to the first three letters if no match is found.&#x20;
* **Heatmap labels too cramped**: increase per‑cell size with `HEATMAP_CELL_INCH` (e.g., `0.65`) and/or reduce tick fonts via `HEATMAP_TICK_FONTSIZE`.&#x20;
* **Bayesian outputs missing**: install `bambi` and `arviz`, or proceed with the frequentist tables only.

---

## 9) Reproducibility notes

* A single global exposure–outcome lag is fixed and documented (the minimal lag script writes a JSON record).
* Standardization parameters (means/SDs) are estimated in‑window and reused for forecasting to avoid leakage.
* All generated CSV/PNGs are placed under `results/` and `visualizations/` with stable filenames.

---

### Quick run (copy–paste)

**macOS/Linux**

```bash
export SUBMISSION_ROOT="$(pwd)"
python 01_run_food2health_v2.py
python 02_build_importance.py
python 03_build_integrated_similarity.py
python make_visuals_oneclick.py
python make_maps_all_in_one.py --th-hub 0.70 --dpi 300
```

**Windows PowerShell**

```powershell
$env:SUBMISSION_ROOT = (Get-Location).Path
python 01_run_food2health_v2.py
python 02_build_importance.py
python 03_build_integrated_similarity.py
python make_visuals_oneclick.py
python make_maps_all_in_one.py --th-hub 0.70 --dpi 300
```

