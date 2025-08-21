# Food–Security to MNBAC Health (GBD 2021, High‑Income Countries)

**End‑to‑end pipeline** to estimate exposure→outcome coefficients, forecast health outcomes, quantify importance and consistency, and visualize cross‑country similarity. The workflow targets 30 high‑income countries with harmonized **2001–2021** history and **2022–2040** projections.

> **Outcomes**: DALYs, deaths, incidence, prevalence
> **Exposures**: 11 FAOSTAT Food‑Security and macro indicators (original units)

---

## 1) What’s in this repository?

* **00\_exposure\_lag\_oneclick\_minimal.py** — One‑click global lag selection (k ∈ {0,…,5}) for exposure→outcome timing.
* **01\_run\_food2health\_v2.py** — Train multi‑output Bayesian ridge per country; produce forecasts & uncertainty.
* **02\_build\_importance.py** — Aggregate coefficients across countries; frequentist consistency + optional Bayesian pooling; export posterior tables.
* **03\_build\_integrated\_similarity.py** — Build “exposure fingerprints” and pairwise cosine similarity; export matrices.
* **make\_visuals\_oneclick.py** — Generate trends, domain mosaics, importance figures (with numeric labels), and the integrated‑similarity heatmap (annotated, lower‑triangle).&#x20;
* **make\_maps\_all\_in\_one.py** — World maps for Hubs, Links, and Top‑Drivers (categorical palettes; discrete link styling).&#x20;

> The visualization scripts above already implement numeric labeling for posterior means/HDIs and P(Coef>0), and an annotated, lower‑triangle cosine heatmap (two decimals).&#x20;

---

## 2) Data requirements

Arrange files under a single project root (called `SUBMISSION_ROOT`). You may keep your project at `D:\GBD MNBAC\Code` (default) or any folder; just point the environment variable there (see Quick start).

**Folder layout (relative to `SUBMISSION_ROOT/`)**:

```
data/
  GBD_by_location_renamed/<Region>/
    Prevalence.csv
    DALYs_(Disability-Adjusted_Life_Years).csv
    Deaths.csv
    Incidence.csv
  ISO/region_abbreviations.xlsx   # flexible column names (Region/Name vs ISO/ISO3/etc.)

results/
  food2health_v2/<Region>/
    UI_2022-2040.csv
    baseline_2022-2040.csv
    HBCR_metrics.csv
  importance/                     # created by 02_... (posterior tables)
  integrated/                     # created by 03_... (fingerprints, similarity)

visualizations/                   # created by the two *make_* scripts
```

**Notes**

* Historical health series are rates and “all ages” to avoid population‑size artifacts (methods consistent across countries).
* `region_abbreviations.xlsx` maps country names to ISO3. Column names are auto‑detected (e.g., `Region/Name/Country` and `ISO/ISO3/Code`).

---

## 3) Software requirements

* **Python** ≥ 3.10
* Core: `numpy`, `pandas`, `matplotlib`
* Optional (recommended):

  * `scikit-learn` (BayesianRidge)
  * `seaborn` (heatmaps)
  * `scipy` (hierarchical ordering)
  * `geopandas` + `pyogrio` (maps) or `geodatasets`
  * `adjustText` (label placement)
  * `bambi` + `arviz` (optional Bayesian pooling in importance)

**Windows virtual environment (PowerShell)**

```powershell
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn seaborn scipy geopandas pyogrio geodatasets adjustText bambi arviz
```

---

## 4) Quick start (Windows, one machine)

1. **Set project root** (where this repo lives):

   ```powershell
   setx SUBMISSION_ROOT "D:\GBD MNBAC\Code"
   ```

   Re‑open the terminal so `SUBMISSION_ROOT` takes effect.

2. **Put data in place** (see §2 layout). Confirm that:

   * `data/GBD_by_location_renamed/<Region>/*.csv` exist for all regions.
   * `data/ISO/region_abbreviations.xlsx` is available.

3. **Run the pipeline in order**:

   ```powershell
   # 0) Global lag selection (writes a small JSON for traceability)
   python 00_exposure_lag_oneclick_minimal.py

   # 1) Fit per-country models; write forecasts & UI to results/food2health_v2/<Region>/
   python 01_run_food2health_v2.py

   # 2) Build importance tables (consistency + optional Bayesian pooling)
   python 02_build_importance.py

   # 3) Build exposure fingerprints & cosine similarity matrices
   python 03_build_integrated_similarity.py

   # 4) Generate all non‑map figures (trends, panels, importance, integrated heatmap)
   python make_visuals_oneclick.py

   # 5) Generate all maps (Hubs, Links, Top‑Driver)
   python make_maps_all_in_one.py --th-hub 0.70 --dpi 300
   ```

   The visuals go to `visualizations/…`. Maps go to `visualizations/maps/`.
   The heatmap is **lower‑triangle, with two‑decimal annotations** out‑of‑the‑box.&#x20;

---

## 5) Script‑by‑script outputs

### 00\_exposure\_lag\_oneclick\_minimal.py

* **Output**: `results/selected_exposure_lag.json`
* Chooses a **single global lag** k ∈ {0,…,5} by minimising the median MAE ratio vs k=0 on the 2017–2021 validation window.

### 01\_run\_food2health\_v2.py

* **Outputs per region**:
  `UI_2022-2040.csv` (main model predictions & UIs)
  `baseline_2022-2040.csv` (history‑only linear baseline)
  `HBCR_metrics.csv` (Adjusted Health‑Burden Change Rate for each outcome)

### 02\_build\_importance.py

* **Outputs**:
  `results/importance/consistency_table.csv` (frequentist summary)
  `results/importance/posterior_indicator.csv` (if `bambi` present)
  `results/importance/prob_positive.csv` (Pr(β>0), if `bambi` present)
* Safe “atomic replace” (writes to a temp folder then renames).

### 03\_build\_integrated\_similarity.py

* **Outputs**:
  `integrated_exposure_fingerprint.csv`, `integrated_exposure_fingerprint_z.csv`
  `aggregated_outcome_AHBCR.csv`
  `pairwise_similarity_cosine.csv` (N×N cosine)
  `pairs_ranked.csv` (unique i\<j pairs by similarity)

### make\_visuals\_oneclick.py (figures)

* **Per‑region trends** (2001–2040) with baseline overlay
* **Domain mosaics**: `visualizations/panels/All_<Domain>_2001-2040.png`
* **AHBCR bubble scatter** + `region_classification_summary.csv`
* **Importance**

  * `posterior_mean_hdi.png` — each row labeled as `mean[lo,hi]` (e.g., `0.80[0.20,0.65]`)
  * `prob_positive.png` — bar labels show P(Coef>0) with 0.05/0.50/0.95 guide lines
* **Integrated similarity**

  * `similarity_heatmap.png` — **lower‑triangle only**, **in‑cell numbers**, optional hierarchical ordering.&#x20;

### make\_maps\_all\_in\_one.py (maps)

* **Hub map** (neighbors above threshold; discrete 0–4 tiers)
* **Links map** (all edges orange‑red; node tiers keep discrete palette)
* **Top‑driver map** (high‑contrast categories; no grey)
  Outputs `hub_map.png`, `hub_legend.png`, `links_map.png`, `links_legend.png`, `top_driver_map.png`, `top_driver_legend.png`.&#x20;

---

## 6) Configuration knobs

* `SUBMISSION_ROOT` (env var): absolute path to this project root.
* **Heatmap labels** (make\_visuals\_oneclick.py):

  * `HEATMAP_TICK_FONTSIZE` (env var, default `9`)
  * `HEATMAP_CELL_INCH` (env var, default `0.50`) controls per‑cell size.&#x20;
* **Maps**: `--th-hub` (default 0.70), `--dpi` (default 300).&#x20;

---

## 7) Reproducibility checklist

* **Python & packages**: list exact versions (`pip freeze > requirements.txt`).
* **Randomness**: BayesianRidge in scikit‑learn is deterministic given inputs; set `numpy.random.seed` if you add stochastic steps.
* **Data timestamp**: record download dates for GBD and FAOSTAT.
* **Lag JSON**: keep `results/selected_exposure_lag.json` with the chosen k.
* **OS**: Windows 10/11 tested; macOS/Linux should work with matching dependencies.

---

## 8) Troubleshooting

* **Geopandas / map errors**:
  Install `geopandas` + `pyogrio`; if `geodatasets` is missing, the map script will try Natural Earth. Firewalls can block auto‑download—install `geodatasets` to use a local copy.&#x20;
* **Heatmap numbers too small/large**:
  Set `HEATMAP_CELL_INCH=0.6` (bigger cells) or reduce `HEATMAP_TICK_FONTSIZE=8`.&#x20;
* **No posterior files**:
  If `bambi`/`arviz` are not installed, importance plots will skip Bayesian outputs; the frequentist table is still produced.
* **ISO mapping**:
  If some labels are wrong, check `data/ISO/region_abbreviations.xlsx` column names and values.

---

## 9) How to cite

> *Project title*. Year. *Food–Security to Health: High‑income countries (GBD 2021)*. GitHub repository: **your‑repo‑URL**.
> Data: IHME GBD 2021 Results Tool; FAOSTAT Food Security (CC BY 4.0).

---

## 10) License

* **Code**: MIT (recommended) — add a `LICENSE` file.
* **Data**: FAOSTAT Food Security database is **CC BY 4.0**; GBD results are subject to IHME terms.

---

## 11) Contributing

Pull requests are welcome. Please:

* Run `flake8`/`black` if you add Python code.
* Include a brief description and a minimal test (e.g., one mock region).
* Do not commit raw proprietary data.

---

## 12) Maintainers / Contact

* Your name — affiliation — email

---

### Appendix A — Repository tree (after running the pipeline)

```
SUBMISSION_ROOT/
  00_exposure_lag_oneclick_minimal.py
  01_run_food2health_v2.py
  02_build_importance.py
  03_build_integrated_similarity.py
  make_visuals_oneclick.py
  make_maps_all_in_one.py
  data/
    GBD_by_location_renamed/...
    ISO/region_abbreviations.xlsx
  results/
    food2health_v2/...        # per‑country outputs
    importance/...
    integrated/...
  visualizations/
    regions/...
    panels/...
    dashboard/...
    importance/...
    integrated_similarity/
    maps/
```

---

### Appendix B — One‑click helpers (optional)

Create a PowerShell script `run_all.ps1` in the repo root:

```powershell
# Set root (edit if needed)
$env:SUBMISSION_ROOT = (Get-Location).Path

python 00_exposure_lag_oneclick_minimal.py
python 01_run_food2health_v2.py
python 02_build_importance.py
python 03_build_integrated_similarity.py
python make_visuals_oneclick.py
python make_maps_all_in_one.py --th-hub 0.70 --dpi 300
```

Run with:

```powershell
.\run_all.ps1
```

---


