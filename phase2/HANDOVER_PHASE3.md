# Phase 2 → Phase 3 handover

Phase 2 is complete. This page lists what Phase 3 receives, how to load it, and the rules that keep the Phase 3
evaluation fair. Paths are relative to the `phase2` folder of the repository.

## 1. What you receive

| What | File | In Git? |
|---|---|---|
| Train / validation / test with the 12 new features (43 columns) | `data/processed/train_v2.csv`, `validation_v2.csv`, `test_v2.csv` | Yes, via Git LFS (190 MB + 41 MB + 41 MB) |
| Preprocessing pipeline, fitted on train | `models/pipeline_v2.pkl` | Yes |
| Names of the 49 pipeline output columns, in order | `results/model_feature_names.csv` | Yes |
| The 9 RFE-selected features | `results/selected_features.csv` | Yes |
| RFE ranking of all 49 features | `results/rfe_ranking.csv` | Yes |
| Feature-engineering code | `src/phase2_features.py` (same code as notebook section 2) | Yes |
| Feature documentation | `results/feature_engineering_summary.csv`, `reports/Phase2_Feature_Engineering_Summary.pdf` | Yes |
| Statistical tests and effect sizes | `results/statistical_tests.csv`, `reports/Phase2_Data_Analysis_Report.pdf` | Yes |
| Data overview and churn by group (train) | `results/column_overview.csv`, `results/churn_rate_by_group.csv` | Yes |
| Highly correlated feature pairs (Pearson or Spearman above 0.7, train) | `results/correlation_pairs.csv` | Yes |
| Baseline model and its metrics | `models/baseline_logreg.pkl`, `results/baseline_metrics.csv` | Yes |
| MLflow runs (params and metrics) | `results/mlflow_runs.csv`; the live runs are in the local MLflow store of whoever runs the notebook | CSV yes, store no |
| Reference data for monitoring (Phase 4) | `monitoring/reference_data.csv`, `monitoring/evidently_train_vs_test.html` | Yes |
| Charts and dashboard | `figures/*.png`, `figures/dashboard.html` | Yes |

Split sizes: 700,000 / 150,000 / 150,000 rows; churn rate 9.92 % in each split (stratified in Phase 1). The target
column is `churn` (0/1); `customer_id` is an identifier, never a feature.

## 2. Getting the processed data

The three `*_v2.csv` files are stored with Git LFS in `phase2/data/processed/`.

1. **From the repository:** install Git LFS, run `git lfs install` once, then clone the repository (or run
   `git lfs pull` in an existing clone). Check that `train_v2.csv` is about 190 MB; a file of a few hundred bytes is
   an LFS pointer, which means LFS was not installed when you cloned.
2. **Regenerate (always works, identical files):** put the Phase 1 outputs in `phase2/data/input/` (or set
   `PHASE1_DIR` in the notebook's Settings cell), `pip install -r requirements.txt`, and run
   `notebooks/02_phase2_code.ipynb`. The files appear in `data/processed/` (`SEED = 42`).

## 3. Loading everything (first cell of Phase 3)

```python
import os, sys, pickle
import pandas as pd

PHASE2_DIR = "../../phase2"            # path from your notebook to the phase2 folder; adjust to your layout
sys.path.insert(0, PHASE2_DIR)
from src.phase2_features import add_features, to_model_features

def p2(*parts):
    return os.path.join(PHASE2_DIR, *parts)

target = "churn"
train = pd.read_csv(p2("data", "processed", "train_v2.csv"))
validation = pd.read_csv(p2("data", "processed", "validation_v2.csv"))
pipeline = pickle.load(open(p2("models", "pipeline_v2.pkl"), "rb"))            # needs sklearn-features
selected = pd.read_csv(p2("results", "selected_features.csv"))["feature"].tolist()

X_train = to_model_features(train, pipeline)          # 700,000 x 49: imputed, scaled, one-hot encoded
X_val = to_model_features(validation, pipeline)       # 150,000 x 49
y_train, y_val = train[target], validation[target]

X_train_9, X_val_9 = X_train[selected], X_val[selected]       # the RFE feature set
```

`to_model_features` runs the already-fitted pipeline (`transform` only; never refit it on validation or test) and
returns a DataFrame with the column names from `results/model_feature_names.csv`. Leave `test_v2.csv` unloaded until the
final evaluation.

**Applying the feature engineering to raw rows** (Phase 1 columns, for example new customers in Phase 4):
`to_model_features(add_features(raw_rows), pipeline)`. `add_features` uses only each customer's own columns (never
`churn`), handles `tenure = 0` and tenure above 72, and leaves missing values for the pipeline's imputers.

## 4. How the features were selected

- RFE (`LogisticRegression(max_iter=1000)`, one feature removed per step) on a **stratified sample of 100,000 training
  rows** (seed 42), after standardizing all 49 pipeline outputs. It ranks all 49 features (`results/rfe_ranking.csv`).
- The number kept was chosen by 5-fold stratified cross-validation on the same sample (`RFECV`, ROC-AUC): the smallest
  number within 0.001 of the best mean score (best 0.6866 at 12 features; 9 features score 0.6860).
- The 9 features: `contract_two_year`, `contract_one_year`, `customer_satisfaction`, `num_service_calls`,
  `num_complaints`, `has_tech_support`, `late_payments`, `num_services`, `has_online_security`.
- Validation and test were not used to rank or choose features.

## 5. The baseline to beat

Logistic Regression (`StandardScaler` + `LogisticRegression(max_iter=1000)`) on the 9 selected features, trained on the
whole training split:

| Metric | 5-fold CV on train | Validation |
|---|---|---|
| ROC-AUC | 0.6841 | **0.6865** |
| PR-AUC | 0.2007 | **0.2015** |
| Precision at 0.5 | — | 0.5270 |
| Recall at 0.5 | — | 0.0026 |
| F1 at 0.5 | — | 0.0052 |

Validation confusion matrix at 0.5: 135,081 true negatives, 35 false positives, 14,845 false negatives, 39 true
positives. A random model's PR-AUC equals the churn rate, 0.0992. Compare threshold-free metrics first (ROC-AUC,
PR-AUC); precision, recall and F1 at 0.5 are reported for completeness and will change once you choose a threshold.

**Reproduce it:** `pickle.load(open(p2("models", "baseline_logreg.pkl"), "rb")).predict_proba(X_val_9)[:, 1]` gives
validation ROC-AUC 0.6865. The CV figures come from a leak-free CV in which preprocessing, scaling and RFE (on a
stratified sample of the training folds) are refitted in every fold (`cv_scores_selected` in the notebook).

## 6. Rules that keep the comparison fair

1. **Test is touched once**, for the final evaluation of the chosen model. Phase 2 only transformed it and compared its
   column distributions with train (Evidently); no model has been trained, tuned or scored on it.
2. **Choose the threshold, hyperparameters and imbalance strategy on train (CV) and validation only.**
3. **SMOTE, if used, goes inside the CV folds and only on the training part** (for example with
   `imblearn.pipeline.Pipeline`). Never resample validation or test.
4. **Same metrics as the baseline**, on the same validation split, so every model is comparable to 0.6865 / 0.2015.
5. **Try both feature sets**: the 9 selected features and all 49. RFE used Logistic Regression; tree models may use
   the other features (the engineered rates and interactions, for example).
6. **Cross-validation and the Phase 2 pipeline:** sklearn's `clone` (used by `cross_val_score`, `GridSearchCV`) fails on
   `DataFrameSelector` from `sklearn-features`. This matters only if you put `pipeline_v2` inside a model being
   cross-validated. Working on the already-transformed `X_train` avoids it; Phase 2's CV used `copy.deepcopy` instead
   of `clone` for the same reason.

## 7. MLflow

Experiment `customer-churn-prediction` (the same name Phase 1 used). Phase 2 runs:

- `phase2-feature-engineering`: `dataset_version = v2-phase2-features`, RFE settings (sample size, selection rule),
  `cv_*` and `val_*` ROC-AUC and PR-AUC for the three feature sets, number of drifted columns; artifacts: pipeline,
  feature lists, reports, Evidently report.
- `phase2-baseline-logistic-regression`: model, features, CV method, threshold; metrics `cv_roc_auc`, `cv_pr_auc`,
  `val_roc_auc`, `val_pr_auc`, `val_precision`, `val_recall`, `val_f1`; artifacts: the model and its metrics table.

MLflow keeps its store (`mlflow.db`, `mlruns/`) in the folder the notebook runs from, and it is not in Git;
`results/mlflow_runs.csv` has the Phase 2 numbers. To see all phases in one MLflow UI, point every phase at the same
tracking location (for example the same `MLFLOW_TRACKING_URI`). Log Phase 3 runs with the same metric names
(`val_roc_auc`, `val_pr_auc`, ...) so they line up.

## 8. Warnings

- Load the pickles with the versions in `requirements.txt` (scikit-learn 1.9.1, sklearn-features 0.0.2).
- `evidently` pins `plotly<6`; keep Plotly 5.24.1 if you install from `requirements.txt` in the same environment.
- Do not overwrite Phase 2 outputs from Phase 3 code; write Phase 3 outputs to the Phase 3 folder.

## 9. Leads from the Phase 2 analysis (to test in Phase 3, not conclusions)

These are associations found on the training data. They suggest what to try; they do not show that a feature causes
churn.

- **`monthlycharges` is non-linear.** Its linear correlation with churn is about −0.01, yet churn falls steadily with
  the monthly charge: 11.3 % up to 50, 10.0 % for 70–90, 8.2 % above 130 (section 3a, `figures/scatter_charges_tenure.png`).
  Tree models can use this directly; for a linear model, compare bands or splines with cross-validation.
- **Near-duplicate features.** `complaints_per_month`, `calls_per_month` and `late_payments_per_month` rank customers
  almost like their counts (Spearman 0.91, 0.72, 0.97); `tenure` and `log_totalcharges` (Spearman 0.96) and
  `has_internet_service` and `addons_without_internet` (−0.92) also overlap. With Logistic Regression, read the
  coefficients of such pairs together, or keep one of each; see `results/correlation_pairs.csv`.
- **Missing values carry almost no signal.** Churn is 9.7–10.2 % where a value is missing vs 9.9 % where it is present
  (`results/column_overview.csv`), so the pipeline's median imputation loses little.
- **All effects are small.** The strongest association is contract type (Cramér's V = 0.139); no feature has
  |r| > 0.09 with churn. Expect gains over the baseline to be modest, and judge them on PR-AUC as well as ROC-AUC.

## 10. Not done in Phase 2 (Phase 3 scope)

Random Forest, XGBoost, SMOTE vs class weights, hyperparameter tuning, threshold selection, and the final test
evaluation. Phase 4 will use `monitoring/reference_data.csv` as the Evidently reference for new data batches.
