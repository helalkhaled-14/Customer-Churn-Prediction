# Phase 2 — Advanced Analysis & Feature Engineering

DEPI / IBM Data Science graduation project: **Customer Churn Prediction**, Milestone 2.

Phase 2 starts from the cleaned and split data that Phase 1 produced and prepares everything that model development
(Phase 3) needs: tested churn drivers, engineered features, an updated preprocessing pipeline, a selected feature
set, a Logistic Regression baseline to beat, a dashboard, and a reference dataset for drift monitoring. It also
explains the data to a newcomer: what each column holds, how many customers churned, and which columns are associated
with churn.

| Requirement (team breakdown, 21 Nov 2026) | Where it is done | Output |
|---|---|---|
| Exploratory analysis: the columns, churned vs stayed, churn by category, numeric distributions, billing vs tenure | Notebook section 3a | `results/column_overview.csv`, `results/churn_rate_by_group.csv`, `figures/churn_distribution.png`, `figures/churn_rate_by_category.png`, `figures/numeric_distributions_by_churn.png`, `figures/scatter_charges_tenure.png` |
| Statistical tests with effect sizes (Cramér's V, Cohen's d) | Section 3b | `results/statistical_tests.csv`, `figures/effect_size_ranking.png` |
| New features and transformations, leakage review per feature | Sections 2 and 5 | `results/feature_engineering_summary.csv`, `src/phase2_features.py`, `models/pipeline_v2.pkl` |
| Correlation analysis (Pearson and Spearman, Cramér's V between categorical columns), and RFE on a stratified sample of the training data | Sections 4 and 6 | `figures/correlation_v2.png`, `results/correlation_pairs.csv`, `figures/categorical_association_v2.png`, `results/rfe_ranking.csv`, `results/selected_features.csv`, `figures/rfe_number_of_features.png` |
| Logistic Regression baseline: preprocessing pipeline, cross-validation, MLflow | Sections 6, 7 and 11 | `models/baseline_logreg.pkl`, `results/baseline_metrics.csv`, MLflow run `phase2-baseline-logistic-regression` |
| Dashboard: churned vs non-churned segmentation and churn trends | Section 8 | `figures/dashboard.html` (8 interactive charts, each with a one-line note) |
| Evidently report, train vs test, with a reference dataset for monitoring | Section 10 | `monitoring/evidently_train_vs_test.html`, `monitoring/reference_data.csv`, `results/evidently_drift_summary.csv` |
| Deliverables: Data Analysis Report, Feature Engineering Summary | `reports/` | two PDFs |

**Handing over to Phase 3:** read [`HANDOVER_PHASE3.md`](HANDOVER_PHASE3.md).

## Folder structure

```
phase2/
├── README.md                      this file
├── HANDOVER_PHASE3.md             what Phase 3 gets and how to start
├── requirements.txt               pinned library versions (the tested ones)
├── .gitignore                     keeps the Phase 1 copies and MLflow's local store out of Git
├── .gitattributes                 stores data/processed/*.csv with Git LFS
├── notebooks/
│   ├── 02_phase2_code.ipynb       the Phase 2 notebook, run from a clean kernel, with outputs
│   └── 02_phase2_explained.ipynb  same code and outputs, with a step-by-step explanation (Arabic)
├── src/
│   └── phase2_features.py         add_features() and the column lists, importable by Phases 3 and 4
├── data/
│   ├── input/                     Phase 1 outputs go here (not in Git): train.csv, validation.csv, test.csv, pipeline.pkl
│   └── processed/                 train_v2.csv, validation_v2.csv, test_v2.csv (Git LFS; also recreated by the notebook)
├── models/
│   ├── pipeline_v2.pkl            preprocessing pipeline (FeatureUnion), fitted on train; 43 columns in, 49 out
│   └── baseline_logreg.pkl        the baseline: StandardScaler + LogisticRegression on the 9 selected features
├── results/
│   ├── column_overview.csv              every column of train_v2: source, group, type, values, missing %, churn % if missing
│   ├── churn_rate_by_group.csv          customers, churned and churn % for each value of the categorical and 0/1 columns
│   ├── statistical_tests.csv            41 tests: statistic, p-value, effect size, Bonferroni, strength, rank
│   ├── correlation_pairs.csv            feature pairs with |r| > 0.7 (Pearson or Spearman), train data
│   ├── feature_engineering_summary.csv  the 12 new features: formula, reason, zero/missing handling, leakage check
│   ├── rfe_ranking.csv                  RFE ranking of all 49 model features
│   ├── selected_features.csv            the 9 features RFE kept
│   ├── model_feature_names.csv          the 49 pipeline output columns, in order
│   ├── baseline_metrics.csv             baseline CV and validation metrics
│   ├── evidently_drift_summary.csv      per-column drift result, train vs test
│   └── mlflow_runs.csv                  params and metrics of the two Phase 2 MLflow runs
├── figures/                       PNG charts from the analysis, and dashboard.html
├── monitoring/
│   ├── reference_data.csv         50,000 stratified training rows (no customer_id), the reference for monitoring
│   └── evidently_train_vs_test.html
└── reports/
    ├── Phase2_Data_Analysis_Report.pdf
    └── Phase2_Feature_Engineering_Summary.pdf
```

## Installing

Python 3.13 was used (3.13.14). From the `phase2` folder:

```bash
pip install -r requirements.txt
```

The first cell of the notebook runs the same command. Notes on the pinned versions:

- `sklearn-features` is required: the Phase 1 and Phase 2 pipelines use its `DataFrameSelector` (the course's Session 8
  pipeline), and the `.pkl` files do not load without it. Load the pickles with the pinned scikit-learn version.
- `evidently==0.7.23` requires `plotly<6`, so Plotly is pinned to 5.24.1. Installing Evidently into an environment that
  has Plotly 6 or newer downgrades Plotly; the dashboard code works with 5.24.1 (tested).

A virtual environment is recommended (`python -m venv .venv`, then activate it). The `.venv` folder is ignored by Git.

## Inputs (from Phase 1)

The notebook reads four Phase 1 outputs and does not change them:

| File | What it is |
|---|---|
| `train.csv` | 700,000 rows, 31 columns, stratified 70 % split |
| `validation.csv` | 150,000 rows |
| `test.csv` | 150,000 rows |
| `pipeline.pkl` | Phase 1 preprocessing pipeline (used only for the "Phase 1 features" comparison) |

Either copy them into `data/input/`, or set `PHASE1_DIR` in the notebook's Settings cell to the folder that holds them
(for example the Phase 1 output folder of the combined repository). Paths are relative, so the notebook runs on
Windows, macOS and Linux. If a file is missing, the Settings cell stops with a message naming it.

## Running

1. Open `notebooks/02_phase2_code.ipynb` (or the explained version; the code is identical) in Jupyter or VS Code.
2. Select the Python environment where you installed `requirements.txt`.
3. **Run All.** On the test laptop the run takes about 5–6 minutes; the slowest steps are the cross-validation and
   the RFE in section 6, and the Spearman correlation in section 4.

Everything is written relative to the `phase2` folder: CSVs to `data/processed/`, the two models to `models/`, tables to
`results/`, charts and the dashboard to `figures/`, the Evidently report and reference data to `monitoring/`. MLflow
writes its local store (`mlflow.db`, `mlruns/`) next to the notebook; both are ignored by Git, and
`results/mlflow_runs.csv` keeps a copy of the runs' parameters and metrics. Re-running gives the same results
(`SEED = 42`).

**Opening the outputs:** `figures/dashboard.html` and `monitoring/evidently_train_vs_test.html` are self-contained pages:
open them in any browser (double-click the file). They work offline. GitHub shows HTML files as source code, so download
them or open them from a local clone.

`src/phase2_features.py` is not imported by the notebook (the notebook is self-contained, as taught in the course). It
is a copy of the notebook's feature-engineering code, checked to produce identical output, for Phases 3 and 4.

## Data

| Data | Where | How to get it |
|---|---|---|
| Phase 1 splits (`train.csv`, `validation.csv`, `test.csv`, `pipeline.pkl`) | Phase 1's folder in the repository | copy into `data/input/` or point `PHASE1_DIR` at them |
| Phase 2 processed splits (`train_v2.csv` 190 MB, `validation_v2.csv` 41 MB, `test_v2.csv` 41 MB) | `phase2/data/processed/`, stored with **Git LFS** | `git lfs install` once, then clone or `git lfs pull`; or re-run the notebook, which recreates identical files |
| Monitoring reference (`reference_data.csv`, 13 MB) | `phase2/monitoring/` | normal Git |

Whoever commits the processed CSVs must run `git lfs install` first; otherwise Git tries to push `train_v2.csv` as a
normal file and GitHub rejects it (100 MB limit). If the team's GitHub LFS quota runs out, the notebook regenerates the
three files in a few minutes from the Phase 1 outputs.

## Key results

- **The data (training split).** 700,000 customers, 69,459 churned (9.92 %); validation and test have the same rate.
  31 Phase 1 columns (an id, 29 features, `churn`) plus 12 Phase 2 features; `results/column_overview.csv` describes
  each one. Nine columns have 2–5 % missing values; churn is almost the same whether a value is missing or not
  (9.7–10.2 % vs 9.9 %).
- **What moves churn.** Contract type is the clear driver: 26.4 % on month-to-month (only 2 % of customers), 12.7 % on
  one-year and 5.7 % on two-year contracts. Tech support (8.6 % with vs 11.3 % without), online security (8.6 % vs
  10.6 %), internet service and tenure group move churn by 1 to 3 points; every other categorical column by less than
  1 point. Churned customers report lower satisfaction (average 5.56 vs 6.22 on the 1–9 scale) and have more
  complaints (0.90 vs 0.68), service calls (2.10 vs 1.72) and late payments (0.49 vs 0.39). Age, credit score,
  tenure and the monthly charge differ very little on average.
- **Billing.** `monthlycharges` is unrelated to `tenure` (r = 0.00) and `totalcharges` follows tenure (r = 0.91). The
  linear correlation of the charges with churn is weak (|r| < 0.02), but churn falls steadily with the monthly
  charge, from 11.3 % (up to 50) to 8.2 % (above 130): a non-linear association that Phase 3 can test.
- **Churn drivers.** 24 of 41 candidate features are significant after the Bonferroni correction, but only 5 have
  more than a negligible effect (all small): contract type, customer satisfaction, number of complaints, number of
  service calls, and the engineered `m2m_x_complaints`. Churn is 26.4 % on month-to-month contracts vs 5.7 % on
  two-year contracts. Demographics show no association with churn.
- **Correlation (train).** The correlation heatmap uses the exact column names; the 11 Phase 2 features in it are
  marked in orange and are all documented in `results/feature_engineering_summary.csv` (for example
  `m2m_x_complaints` = `num_complaints` for month-to-month customers, else 0). Seven feature pairs exceed |r| = 0.7
  (`results/correlation_pairs.csv`): `tenure`/`log_totalcharges` and `has_internet_service`/`addons_without_internet`
  by both methods; the three per-month rates and their counts by Spearman only (0.72–0.97, same ranking, different
  scale); `log_avg_monthly_gb` and the internet columns by Pearson only. No feature has |r| > 0.09 with churn. The
  categorical columns are independent of one another (Cramér's V ≤ 0.003).
- **Feature selection.** RFE on a stratified 100,000-row training sample ranked all 49 features; 5-fold
  cross-validation on the same sample chose **9 features** (the smallest number within 0.001 of the best CV ROC-AUC,
  0.6866). On validation, 9 features score ROC-AUC 0.6865 vs 0.6869 for all 49 and 0.6869 for the Phase 1 features.
  The engineered features did not help Logistic Regression; tree models in Phase 3 may still use them.
- **Baseline to beat (validation):** ROC-AUC **0.6865**, PR-AUC **0.2015** (a random model scores 0.0992, the churn
  rate). Leak-free 5-fold CV on train: ROC-AUC 0.6841, PR-AUC 0.2007. At the default 0.5 threshold the model flags
  almost nobody (recall 0.0026), because its highest predicted probability is 0.70 and churn is rare; choosing a
  threshold is Phase 3 work.
- **Train vs test drift (Evidently 0.7.23):** no drift in any of the 42 columns. The largest drift score is 0.015
  (`avg_monthly_gb`, Wasserstein distance) against a threshold of 0.1. This is expected from Phase 1's stratified
  random split.

## Method notes

- Statistical tests, fitting, scaling, correlation and feature selection use the training split only. Validation is
  used only to compare the three feature sets and to report the baseline; it plays no part in choosing features. The
  test split is transformed and saved for Phase 3, and its column distributions are compared with train in the
  Evidently report; no model is trained, tuned or scored on it.
- RFE runs on a stratified sample of 100,000 training rows (`train_test_split` with `stratify`, seed 42), as the team
  breakdown specifies. The sample keeps the 9.92 % churn rate. The number of features is chosen with `RFECV` (5-fold
  stratified CV, ROC-AUC) on the same sample.
- Cross-validation is leak-free: preprocessing, scaling and RFE (on a stratified sample of the training folds) are
  refitted inside every fold, and the model is trained on the training folds only.
- Effect sizes use Cohen's thresholds (d: 0.2 / 0.5 / 0.8; V: 0.1 / 0.3 / 0.5). Drivers are ranked on one scale, the
  absolute correlation with churn (Cramér's V or point-biserial r).
- Correlation: Pearson (for two 0/1 columns it equals phi; for a 0/1 and a numeric column, point-biserial r) and
  Spearman (rank-based, for monotonic but non-linear relations and skewed counts). Categorical columns are compared
  with Cramér's V, not with a correlation coefficient.
- All of these are associations in this dataset. None of them shows that a feature causes churn.

## Known limitations and dependencies

- **Phase 1 dependency.** Phase 2 needs the four Phase 1 files above. If Phase 1 changes its split or pipeline,
  re-run this notebook; every Phase 2 output is regenerated.
- **Pickles are version-bound.** `pipeline_v2.pkl` and `baseline_logreg.pkl` need `sklearn-features` and the pinned
  scikit-learn.
- **The baseline is weak by design.** It is the reference, not the model to deploy. Class imbalance (9.92 % churn) and
  the decision threshold are not handled in Phase 2.
- **MLflow store is local.** Each person who runs the notebook gets their own `mlflow.db`; `results/mlflow_runs.csv` is
  the shared record of the Phase 2 runs.
- **Evidently scope.** The team breakdown puts the train-vs-test Evidently report in Phase 2 (done here). The
  requirements document lists Evidently under Week 4 monitoring; Phase 4 builds on `monitoring/reference_data.csv`.

## Tested on

| Computer | What was run | Python and libraries |
|---|---|---|
| Windows 11 laptop | Both notebooks from a fresh kernel in this folder layout, then a separate QA notebook | Python 3.13.14; the versions in `requirements.txt` |

27 Nov 2026 (after the analysis review): each notebook was run top to bottom in a new kernel started with
`jupyter_client` (VS Code would not start a notebook kernel on the test laptop that day), the explained notebook first
(379 s) and the code notebook last (277 s), with no errors. Both produced byte-identical data, models, tables and PNG
charts; only `dashboard.html` and the Evidently report (random element ids) and `results/mlflow_runs.csv` (new run ids,
same parameters and metrics) differ between runs. A separate QA notebook then re-checked the folder, including the new
outputs: every chart label is a real column, the new tables match a fresh computation, and the dashboard has its 8
charts in the right order. The earlier full QA (26 Nov) passed all 186 checks, and `pip check` reported no broken requirements; the same
notebook run on Linux gave the same results (numeric differences below 1e-11).

## Changes on 27 Nov 2026 (analysis review)

A review of the Phase 2 analysis; the split, the feature engineering, the RFE method and the baseline are unchanged.

| Type | Change |
|---|---|
| Error fixed | Dashboard trend chart: tenure groups were sorted as text, so each line went 0-6, 13-24, 25-48, 49+ and back to 7-12. They now run in time order. |
| Checked, no error | The correlation heatmap uses exact column names. `m2m_x_complaints` (shown as "M2M x Complaints" in a teammate's note) is a real, documented Phase 2 feature: `num_complaints` for month-to-month customers, else 0. The heatmap now marks the 11 Phase 2 features in orange, and the notebook checks every label against the data. |
| Added analysis | Section 3a (what the data contains, churned vs stayed, churn by category, numeric distributions, billing vs tenure scatter plots, missing values vs churn); Spearman correlation and Cramér's V between categorical columns in section 4; two dashboard charts (churned vs stayed, churn rate of every category and service) and a short data-based note under every chart. |
| Documentation | `results/feature_engineering_summary.csv` now says which features inherit missing values and are filled with the training median by the pipeline. |
| Reports | Data Analysis Report: new section 3.1 (the data at a glance), Spearman and Cramér's V in 3.5, the new charts and the trend fix in section 5, the monthly-charge finding, and association-not-causation wording. Feature Engineering Summary: the missing-value notes, the overlap of the per-month rates with their counts, and the processed CSVs now said to be in Git LFS (they were described as not in Git). |

