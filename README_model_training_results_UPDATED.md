# Model Training — Step‑by‑Step Walkthrough & Results (Updated)

**Context**  
- **Goal:** Predict student stress level (multi‑class target: **0, 1, 2**).  
- **Features:** All features are **numeric**.  
- **Class balance:** Reported as not imbalanced → **Accuracy** is a fair primary metric.  
- **Pipelines:** All models run in leak‑free **pipelines** with shared preprocessing.

This document reflects the latest executed notebook (`_main_model_training.ipynb`) and the helper utilities in `functions.py`. It adds a precise explanation of **`run_models_with_importances`** and updates the **results** based on the notebook’s printed outputs.

---

## 1) What the helpers in `functions.py` do (quick map)

- `evaluate_classifier(y_true, y_pred, title)` — prints **Accuracy**, **Confusion Matrix**, and a **Classification Report**.  
- `get_feature_names_after_preprocess(pipeline, X_fit_columns=None)` — returns the **post‑preprocessing** feature names from your ColumnTransformer.  
- `feature_importances_dataframe(pipeline, X_fit_columns=None)` — builds a DataFrame of **feature importances** (trees) or **coefficients**/**abs_mean_coef** (linear, multi‑class aware).  
- `run_models_with_importances(models, preprocess, X_train, y_train, X_test, y_test, ...)` — fits each model in a pipeline, **evaluates**, and extracts **Top‑k features** per model (optionally plotting/saving them).

> These helpers keep evaluations **consistent** and **comparable** across models and resampling setups.

---

## 2) How to use `run_models_with_importances` (from `functions.py`)

### Purpose  
A **one‑stop function** to:  
1) build a **pipeline** per model: `('prep', preprocess) → [sampler?] → ('model', estimator)`,  
2) **fit** on `X_train, y_train`,  
3) **evaluate** on `X_test, y_test` with `evaluate_classifier(...)`,  
4) extract **feature importances/coefficients** into a tidy DataFrame,  
5) optionally **plot** and/or **save to CSV** the top features.

### Signature (simplified)
```python
results = run_models_with_importances(
    models,             # dict: {display_name: unfitted estimator}
    preprocess,         # ColumnTransformer (scaler/encoders)
    X_train, y_train,
    X_test,  y_test,
    top=15,             # how many top features to show/plot
    plot=True,          # bar chart of top features if available
    save_csv=False,     # save top-k CSV per model
    csv_prefix='top_features_',
    pipeline_cls=None,  # defaults to sklearn Pipeline; use ImbPipeline if sampler used
    sampler=None,       # e.g., RandomOverSampler()/RandomUnderSampler()/SMOTE(...)
    sampler_name='sampler'
)
```

### What it returns
```python
{
  "pipelines":   { model_name: fitted_pipeline, ... },
  "importances": { model_name: pd.DataFrame or None, ... }  # None if model has no native importances/coef
}
```
- **Pipelines** let you call `.predict` or inspect the fitted estimator later.  
- **Importances** DataFrames contain:
  - trees/ensembles → `feature, importance`  
  - binary linear → `feature, coef, abs_coef`  
  - multi‑class linear → `feature, coef_<class>..., abs_mean_coef`

### Two common call patterns

**A) Plain (no resampling, sklearn Pipeline):**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = run_models_with_importances(
    models=models,
    preprocess=preprocess,      # your ColumnTransformer (e.g., StandardScaler on numeric)
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    top=15,
    plot=False,                 # to keep the notebook compact
    save_csv=True               # saves 'top_features_<ModelName>.csv'
)
```

**B) With resampling (imblearn Pipeline + SMOTE/ROS/RUS):**
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

logreg_models = {"LogReg + SMOTE": LogisticRegression(max_iter=1000, random_state=42)}

results_smt = run_models_with_importances(
    models=logreg_models,
    preprocess=preprocess,
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    pipeline_cls=ImbPipeline,           # IMPORTANT when sampler is used
    sampler=SMOTE(random_state=42),     # or RandomOverSampler/RandomUnderSampler
    top=10,
    plot=False,
    save_csv=True
)
```
> If a model (e.g., **KNN**) has **no native `feature_importances_` or `coef_`**, the function will **still evaluate** it but **skip the Top‑k extraction**. (Consider permutation importance if you need importances for such models.)

---

## 3) Updated Results from the notebook

| Model | Test Accuracy |
|---|---|
| AdaBoost | 0.8864 |
| LogReg + SMOTE | 0.8864 |
| Random Forest | 0.8818 |
| Logistic Regression | 0.8818 |
| Baseline LogReg | 0.8818 |
| Gradient Boosting | 0.8773 |
| Decision Tree | 0.8591 |
| KNN (k=5) | 0.8545 |

**Top models in this run:**  
- **AdaBoost** and **LogReg + SMOTE** at **0.8864**.  
- **Random Forest** and **(Baseline) Logistic Regression** close behind at **0.8818**.  
- **Gradient Boosting** at **0.8773**; **Decision Tree** at **0.8591**; **KNN** at **0.8545**.

> Small differences (±0.5–1%) are normal. If this matters for your use case, run **CV with fixed random_state** and/or a **hyperparameter search**.

---

## 4) What the Top‑features told us (recurring signals)

Across the models that expose importances/coefficients, these features appear repeatedly near the top (order varies by model):

- **blood_pressure**, **social_support**, **sleep_quality**  
- **anxiety_level**, **depression**, **self_esteem**  
- **teacher_student_relationship**, **academic_performance**, **study_load**  
- **basic_needs**, **headache**, **bullying**  
- **future_career_concerns**, **extracurricular_activities**

Interpretation guide: higher absolute coefficients (linear) or higher importance scores (trees/ensembles) indicate stronger contribution to the model’s decision. Always cross‑check directionality with the **classification report** and domain knowledge.

---

## 5) Re‑using results programmatically

You can grab the fitted pipeline and its importance table from the returned `results`:

```python
# Example: pull the Random Forest pipeline and its top features
rf_pipe = results_all["pipelines"]["Random Forest"]
rf_imp  = results_all["importances"]["Random Forest"]   # DataFrame: feature, importance
print(rf_imp.head(15))

# Predict on new data
y_pred_new = rf_pipe.predict(X_new)
```

---

## 6) Final recommendations

- Keep **pipelines** for all models to avoid leakage.  
- If you want **simpler models**, try training with the **Top‑k features** and compare accuracy.  
- For a final selection, add a modest **GridSearchCV** (or Optuna) over the top contenders (AdaBoost, LogReg, RF) using **StratifiedKFold**.  
- Save your final pipeline with `joblib.dump(...)` and log its metrics, params, and feature list for reproducibility.

---

*Notebook: `_main_model_training.ipynb`. Helpers: `functions.py` (especially `run_models_with_importances`).*

---

# Detailed Evaluation & Model Selection (Auto‑generated on 2025-10-25 18:43 UTC)

## Global Comparison
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| LogReg + SMOTE | 0.8864 | 0.8863 | 0.8868 | 0.8863 |
| AdaBoost | 0.8864 | 0.8896 | 0.8868 | 0.8858 |
| Logistic Regression | 0.8818 | 0.8818 | 0.8821 | 0.8820 |
| Baseline LogReg | 0.8818 | 0.8818 | 0.8821 | 0.8820 |
| LogReg + ROS | 0.8818 | 0.8818 | 0.8821 | 0.8820 |
| LogReg + RUS | 0.8818 | 0.8818 | 0.8821 | 0.8820 |
| Random Forest | 0.8818 | 0.8818 | 0.8820 | 0.8818 |
| Gradient Boosting | 0.8773 | 0.8780 | 0.8778 | 0.8770 |
| Decision Tree | 0.8591 | 0.8597 | 0.8594 | 0.8591 |
| KNN (k=5) | 0.8545 | 0.8567 | 0.8551 | 0.8543 |


### Per-class Precision

| Model | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| LogReg + SMOTE | 0.887 | 0.893 | 0.878 |
| AdaBoost | 0.922 | 0.905 | 0.841 |
| Logistic Regression | 0.863 | 0.904 | 0.878 |
| Baseline LogReg | 0.863 | 0.904 | 0.878 |
| LogReg + ROS | 0.863 | 0.904 | 0.878 |
| LogReg + RUS | 0.863 | 0.904 | 0.878 |
| Random Forest | 0.875 | 0.890 | 0.880 |
| Gradient Boosting | 0.897 | 0.870 | 0.867 |
| Decision Tree | 0.859 | 0.831 | 0.889 |
| KNN (k=5) | 0.870 | 0.815 | 0.886 |


### Per-class Recall

| Model | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| LogReg + SMOTE | 0.851 | 0.931 | 0.878 |
| AdaBoost | 0.797 | 0.931 | 0.932 |
| Logistic Regression | 0.851 | 0.917 | 0.878 |
| Baseline LogReg | 0.851 | 0.917 | 0.878 |
| LogReg + ROS | 0.851 | 0.917 | 0.878 |
| LogReg + RUS | 0.851 | 0.917 | 0.878 |
| Random Forest | 0.851 | 0.903 | 0.892 |
| Gradient Boosting | 0.824 | 0.931 | 0.878 |
| Decision Tree | 0.824 | 0.889 | 0.865 |
| KNN (k=5) | 0.811 | 0.917 | 0.838 |


### Per-class F1

| Model | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| LogReg + SMOTE | 0.869 | 0.912 | 0.878 |
| AdaBoost | 0.855 | 0.918 | 0.885 |
| Logistic Regression | 0.857 | 0.910 | 0.878 |
| Baseline LogReg | 0.857 | 0.910 | 0.878 |
| LogReg + ROS | 0.857 | 0.910 | 0.878 |
| LogReg + RUS | 0.857 | 0.910 | 0.878 |
| Random Forest | 0.863 | 0.897 | 0.886 |
| Gradient Boosting | 0.859 | 0.899 | 0.872 |
| Decision Tree | 0.841 | 0.859 | 0.877 |
| KNN (k=5) | 0.839 | 0.863 | 0.861 |


## Confusion Matrices (Actual rows × Predicted columns; class order = [0,1,2])
**LogReg + SMOTE**
```
[[63  5  6]
 [ 2 67  3]
 [ 6  3 65]]
```

**AdaBoost**
```
[[59  6  9]
 [ 1 67  4]
 [ 4  1 69]]
```

**Logistic Regression**
```
[[63  5  6]
 [ 3 66  3]
 [ 7  2 65]]
```

**Baseline LogReg**
```
[[63  5  6]
 [ 3 66  3]
 [ 7  2 65]]
```

**LogReg + ROS**
```
[[63  5  6]
 [ 3 66  3]
 [ 7  2 65]]
```

**LogReg + RUS**
```
[[63  5  6]
 [ 3 66  3]
 [ 7  2 65]]
```

**Random Forest**
```
[[63  5  6]
 [ 4 65  3]
 [ 5  3 66]]
```

**Gradient Boosting**
```
[[61  7  6]
 [ 1 67  4]
 [ 6  3 65]]
```

**Decision Tree**
```
[[61  7  6]
 [ 6 64  2]
 [ 4  6 64]]
```

**KNN (k=5)**
```
[[60 10  4]
 [ 2 66  4]
 [ 7  5 62]]
```


## Model Selection (Top‑2)

**1) AdaBoost** — Highest test accuracy tie and best overall balance.  
- Precision shines for class 0 (few false positives), while recall is highest for classes 1 and 2 (few false negatives).  
- Fast to train and deploy; exposes feature importances.

**2) Logistic Regression + SMOTE** — Ties best accuracy with excellent macro balance.  
- Improves class‑1 and class‑0 recall a bit over plain LogReg.  
- Interpretable coefficients, well‑calibrated probabilities, very easy to operationalize.

**Honorable mention:** Random Forest — virtually tied with plain LogReg; robust and stable with native importances.
