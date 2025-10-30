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
- `knn_rank_with_confidence(...)` — evaluates KNN across k, computes **CV mean ± 95% CI**, and recommends the **k with the largest lower CI bound**. :contentReference[oaicite:5]{index=5}

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
    models,                 # dict: {display_name: unfitted estimator}
    preprocess,             # ColumnTransformer (scaler/encoders)
    X_train, y_train,
    X_test,  y_test,
    top=15,                 # how many top features to show/plot
    plot=True,              # bar chart of top features if available
    save_csv=False,         # save top-k CSV per model
    csv_prefix='top_features_',
    pipeline_cls=None,      # defaults to sklearn Pipeline; use ImbPipeline if sampler used
    sampler=None,           # e.g., RandomOverSampler()/RandomUnderSampler()/SMOTE(...)
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

| Model                                 | Test Accuracy |
|---                                    |---:           |
| Random Forest                         | 0.8955        |
| Gradient Boosting + hyperparams       | 0.8955        |
| Logistic Regression + RUS             | 0.8909        |
| Logistic Regression + SMOTE           | 0.8909        |
| Logistic Regression                   | 0.8864        |
| Logistic Regression + ROS             | 0.8864        |
| Random Forest + hyperparams           | 0.8818        |
| AdaBoost + hyperparams                | 0.8818        |
| AdaBoost                              | 0.8773        |
| Gradient Boosting                     | 0.8591        |
| Decision Tree                         | 0.8545        |
| KNN (k=17, CI-LCB)                    | 0.8545        |

**Top models in this run:**  
- **Random Forest** and **Gradient Boosting (tuned)** — tied for **highest** test accuracy (**0.8955**).  
- Close contenders: **Logistic Regression + RUS/SMOTE** at **0.8909**.

> Small differences (±0.5–1%) are normal; for high-stakes selection, confirm with **CV** and/or a **small hyperparameter search** on the finalists.

---

## 4) What the Top‑features told us (recurring signals)

Across models that expose importances/coefficients (trees/ensembles and linear), the following repeatedly appear near the top (order varies by model):

**blood_pressure, sleep_quality, social_support, anxiety_level, depression, self_esteem, academic_performance, study_load, future_career_concerns, bullying**.  

- **Directionality (linear):** coefficients often show that **self_esteem ↑ → stress ↓**; **bullying/anxiety/depression ↑ → stress ↑**; **sleep_quality ↑ → stress ↓**; **study_load/future_career_concerns ↑ → stress ↑**.  
- **Importance (trees):** these features carry most split gain, confirming their predictive signal.  
*(Use `feature_importances_dataframe(...)` + `plot_top_importances(...)` to see model-specific rankings.)*  :contentReference[oaicite:5]{index=5}

> **Interpretation guide:** higher absolute coefficients (linear) or higher importance scores (trees/ensembles) indicate stronger contribution to the model’s decision. Always cross‑check directionality with the **classification report** and domain knowledge.

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

# Detailed Evaluation & Model Selection

## Global Comparison
| Model               | Accuracy  | Macro Precision | Macro Recall  | Macro F1 |
|---                  |---:       |---:             |---:           |---:      |
| LogReg + SMOTE      | 0.8864    | 0.8863          | 0.8868        | 0.8863 |
| AdaBoost            | 0.8773    | 0.8809          | 0.8776        | 0.8774 |
| Logistic Regression | 0.8864    | 0.8888          | 0.8865        | 0.8873 |
| Baseline LogReg     | 0.8864    | 0.8888          | 0.8865        | 0.8873 |
| LogReg + ROS        | 0.8864    | 0.8880          | 0.8865        | 0.8871 |
| LogReg + RUS        | 0.8909    | 0.8933          | 0.8910        | 0.8918 |
| Random Forest       | 0.8955    | 0.8960          | 0.8956        | 0.8953 |
| Gradient Boosting   | 0.8591    | 0.8598          | 0.8594        | 0.8596 |
| Decision Tree       | 0.8545    | 0.8558          | 0.8551        | 0.8541 |
| KNN (k=17)          | 0.8545    | 0.8693          | 0.8552        | 0.8560 |


### Per-class Precision

| Model               | Class 0 | Class 1 | Class 2 |
|---                  |---:|---:|---:|
| LogReg + SMOTE      | 0.8421 | 0.8649 | 0.8533 |
| AdaBoost            | 0.8939 | 0.9296 | 0.8193 |
| Logistic Regression | 0.8421 | 0.9559 | 0.8684 |
| Baseline LogReg     | 0.8421 | 0.9559 | 0.8684 |
| LogReg + ROS        | 0.8421 | 0.9420 | 0.8800 |
| LogReg + RUS        | 0.8442 | 0.9559 | 0.8800 |
| Random Forest       | 0.9130 | 0.8919 | 0.8831 |
| Gradient Boosting   | 0.8267 | 0.9014 | 0.8514 |
| Decision Tree       | 0.8806 | 0.8354 | 0.8514 |
| KNN (k=17)          | 0.9516 | 0.7444 | 0.9118 |


### Per-class Recall

| Model               | Class 0 | Class 1 | Class 2 |
|---                  |---:|---:|---:|
| LogReg + SMOTE      | 0.8649 | 0.9167 | 0.8919 |
| AdaBoost            | 0.7973 | 0.9167 | 0.9189 |
| Logistic Regression | 0.8649 | 0.9028 | 0.8919 |
| Baseline LogReg     | 0.8649 | 0.9028 | 0.8919 |
| LogReg + ROS        | 0.8649 | 0.9028 | 0.8919 |
| LogReg + RUS        | 0.8784 | 0.9028 | 0.8919 |
| Random Forest       | 0.8514 | 0.9167 | 0.9189 |
| Gradient Boosting   | 0.8378 | 0.8889 | 0.8514 |
| Decision Tree       | 0.7973 | 0.9167 | 0.8514 |
| KNN (k=17)          | 0.7973 | 0.9306 | 0.8378 |


### Per-class F1

| Model               | Class 0 | Class 1 | Class 2 |
|---                  |---:|---:|---:|
| LogReg + SMOTE      | 0.8533 | 0.9362 | 0.8859 |
| AdaBoost            | 0.8429 | 0.9231 | 0.8662 |
| Logistic Regression | 0.8533 | 0.9286 | 0.8800 |
| Baseline LogReg     | 0.8533 | 0.9286 | 0.8800 |
| LogReg + ROS        | 0.8533 | 0.9220 | 0.8859 |
| LogReg + RUS        | 0.8609 | 0.9286 | 0.8859 |
| Random Forest       | 0.8811 | 0.9041 | 0.9007 |
| Gradient Boosting   | 0.8322 | 0.8951 | 0.8514 |
| Decision Tree       | 0.8369 | 0.8745 | 0.8514 |
| KNN (k=17)          | 0.8676 | 0.8272 | 0.8732 |


## Confusion Matrices (Actual rows × Predicted columns; class order = [0,1,2])
**LogReg + SMOTE**
```python
[64  3  7]   # Class 0: a few 0→2
[ 4 66  2]   # Class 1: strong diagonal
[ 8  0 66]   # Class 2: strong diagonal; some 2→0
```
> **Class 1/2 recall** is excellent (66/72, 66/74). Good pick if minimizing **missed stressed cases** is your priority

**AdaBoost**
```python
[59  4 11]   # Class 0: more 0→2 mistakes
[ 2 66  4]   # Class 1: strong diagonal
[ 5  1 68]   # Class 2: strong diagonal
```
> **Excellent recall** for Classes **1** and **2** (66/72, 68/74), but **Class-0 recall** drops (59/74). Tends to confuse **0 → 2** (11 cases), which hurts overall macro in this run

**Logistic Regression**/**Baseline LogReg**
```python
[64  3  7]
[ 4 65  3]
[ 8  0 66]
```
> Very stable and interpretable. Slightly lower Class-1 recall vs SMOTE, but great precision for Class-1.

**LogReg + ROS**
```python
[64  4  6]
[ 4 65  3]
[ 8  0 66]
```

**LogReg + RUS**
```python
[65  3  6]
[ 4 65  3]
[ 8  0 66]
```
> Slight gains on **Class 0** recall (65/74) vs other LR variants; **Class 2** still very good (66/74). Mislabels for 0→2 are present but controlled. Strong macro balanc

**Random Forest**
```python
[63  6  5]
[ 2 66  4]
[ 4  2 68]
```
> Very **balanced**: high diagonal across all rows; small, spread errors. This matches its **high macro F1** and best accuracy.

**Gradient Boosting**
```python
[62  5  7]
[ 4 64  4]
[ 9  2 63]
```
> Lower diagonal than RF, especially **Class-2 (63/74)**, explaining its lower global numbers. (Note: your **tuned** GB in the updated results table ties RF at 0.8955, but the per-class tables reflect the **baseline** GB.)

**Decision Tree**
```python
[59  8  7]
[ 2 66  4]
[ 6  5 63]
```
> More off-diagonal mass → classic **variance/overfit** behavior relative to ensembles.

**KNN (k=17)**
```python
[59 12  3]   # 0→1 inflation
[ 2 67  3]   # Good 1 recall, but…
[ 1 11 62]   # 2→1 inflation
```
> Over-predicts **Class-1** (see 0→1 and 2→1), which matches **high Class-1 recall but low Class-1 precision**.** Not as balanced

## Model Selection (Top‑3)

## Random Forest — Best overall balance

  - **Why:** Highest **Accuracy (0.8955)** and best **Macro F1 (0.8953)** in the global table; per-class F1s are all high and tight (C0 **0.881**, C1 **0.904**, C2 **0.901**); confusion matrix is the most **diagonal-heavy** across all three classes.
  - **When to pick:** You want balanced errors across 0/1/2, strong generalization, and native feature importances. 

## Logistic Regression + RUS — Simplicity with near-SOTA performance
  - **Why:** Accuracy **0.8909** with **Macro F1 0.8918**, the best among LR variants; gains a bit of **Class-0 recall** while keeping **Class-1/2** recall very strong; super **interpretable** and **lightweight** for deployment.
  - **When to pick:** You need **interpretability** and minimal complexity with performance within ~0.5% of the top.

## Logistic Regression + SMOTE — Prioritize catching stressed students
  - **Why:** Ties near the top in Accuracy (**0.8909**) with the best **Class-1/2 recall** among LR variants. If recall on **stressed classes (1/2)** is your north star, SMOTE’s confusion matrix shows fewer **false negatives** there.
  - **When to pick:** You prefer **slightly higher recall** on stressed classes even if macro precision dips a little.

> **Note on Gradient Boosting (tuned):** the **updated results table** shows **GB+HP** tying Random Forest at **0.8955** accuracy. If that tuned model is the candidate for deployment (and reproducibly re-fit on the same schema), it can **swap into #1/#2** with Random Forest. Use the tuned model’s per-class tables on the same test split to confirm macro balance before finalizing

> “**Random Forest** is the most balanced and accurate overall; **LogReg+RUS** matches it closely with simpler, more interpretable deployment; and **LogReg+SMOTE** wins when recall on stressed students is the north star.”

---

## Authors:

- **Luis Pablo Aiello** — Data Analytics Student (Cohort Sep-2025)

---

## License
Educational use within the bootcamp cohort; dataset is survey-based and used for learning purposes.
