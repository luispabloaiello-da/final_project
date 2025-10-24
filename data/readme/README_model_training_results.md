# Model Training — Step‑by‑Step Walkthrough & Results

**Context**  
- **Goal:** Predict student stress level (multi‑class target: **0, 1, 2**).  
- **Data shape:** **All features are numeric** (no categorical columns).  
- **Class balance:** Stated as **not imbalanced**, so **Accuracy** is a fair primary metric.  
- **Style:** Everything runs through **pipelines** so preprocessing is leak‑free and reproducible.

This README explains **what each cell does**, **why it was used**, and **what the key parameters mean**. It finishes with the **results table** and **plain‑English conclusions**, and shows how to **use the Top‑15 features** to simplify a model without losing much accuracy.

---

## 1) Imports & Helper Functions

You import common scientific Python libraries and bring in a set of **project helpers** from `functions.py`. Those helpers provide:

- `evaluate_classifier(y_true, y_pred, title)` → prints **Accuracy**, a **Confusion Matrix**, and a detailed **Classification Report** (precision/recall/F1 for each class). fileciteturn7file0  
- `feature_importances_dataframe(pipeline, X_fit_columns)` → aligns **post‑preprocessing feature names** with either **tree importances** or **linear coefficients** (multi‑class aware). fileciteturn7file0  
- `run_models_with_importances(...)` → convenience wrapper to fit models in **pipelines**, evaluate them with `evaluate_classifier`, and display **Top features** per model. fileciteturn7file0

> Why helpers? They avoid repetitive code and make evaluations **consistent and comparable** across models.

---

## 2) Load Data & Choose the Target

- You select the target column (stress level: **0/1/2**) and define `X` (features) and `y` (target).
- Since you reported **no class imbalance**, we’ll use **accuracy** as the main score, but still look at class‑wise metrics via the classification report for sanity.

> **Why split the data?** We want a **hold‑out** test set that simulates new, unseen data.

---

## 3) Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `test_size=0.2` → 80% train, 20% test.  
- `random_state=42` → reproducible split.  
- `stratify=y` → keeps the **class proportions** consistent in train and test (good practice for multi‑class).

---

## 4) Preprocessing (Numeric‑only)

Because you have **numeric features only**, preprocessing is simple:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_cols = X.columns  # all numeric
preprocess = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_cols)],
    remainder='drop'  # nothing else to transform
)
```

- **`StandardScaler`** rescales each numeric feature to zero mean / unit variance.  
  - Helps **Logistic Regression** and **KNN** (distance‑based) perform better and converge faster.  
  - Trees (Decision Tree, Random Forest, Gradient Boosting) **don’t need scaling**, but including the scaler in one shared pipeline is clean and consistent.

> Everything is used through a **Pipeline** so preprocessing is learned on **train only** (no leakage) and applied identically at test time.

---

## 5) Pipelines

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([('prep', preprocess), ('model', SomeEstimator(...))])
```

- **Why Pipeline?** Guarantees the **same preprocessing** is applied in CV/training and in final predictions; avoids subtle bugs and leakage.

---

## 6) Baseline Models (what, why, key params)

### 6.1 Logistic Regression (multi‑class)
- **What:** Learns a **linear** boundary that separates classes; outputs probabilities.  
- **Why:** Strong, simple **baseline**; fast and interpretable.  
- **Params you used/should know:**
  - `max_iter=1000` → let the optimizer converge reliably.  
  - `multi_class='auto'` → scikit chooses OvR or multinomial; both fine here.  
  - (Optional) `C` = inverse regularization (smaller `C` = stronger penalty).  
- **Needs scaling?** **Yes** — done by `StandardScaler` in the pipeline.

### 6.2 KNN (k‑Nearest Neighbors)
- **What:** Classifies by **voting** among the **k closest** training points.  
- **Why:** Good sanity‑check baseline; no training computation (lazy learner).  
- **Key params:** `n_neighbors=5` is typical; try a small range (3–11).  
- **Needs scaling?** **Absolutely.**

### 6.3 Decision Tree
- **What:** Series of **if/else** splits; captures interactions and non‑linear patterns.  
- **Why:** Interpretable structure; often weaker than ensembles alone.  
- **Key params:** `max_depth`, `min_samples_split`, `min_samples_leaf` (control overfitting).  
- **Needs scaling?** No.

---

## 7) Stronger Models (Ensembles)

### 7.1 Random Forest (bagging)
- **What:** Many trees trained on random subsets; results **averaged** (robust).  
- **Key params:**  
  - `n_estimators` (e.g., 300) → more trees = more stable.  
  - `max_depth` → cap tree depth to reduce overfitting.  
  - `max_features` → features tried at each split (e.g., 'sqrt').

### 7.2 Gradient Boosting (boosting)
- **What:** Builds trees **sequentially**; each new tree fixes errors of the previous ones.  
- **Key params:**  
  - `n_estimators` (number of boosting stages),  
  - `learning_rate` (how strongly each tree contributes),  
  - `max_depth` (depth of each small tree, usually 2–3 for tabular).

> Ensembles usually outperform single models because they **reduce variance** (RF) or **sequentially correct mistakes** (GB).

---

## 8) Evaluation Metrics (why these)

Your helper `evaluate_classifier(...)` prints: **Accuracy**, **Confusion Matrix**, and **Classification Report** (per‑class precision/recall/F1). With **balanced classes**, **Accuracy** is a reasonable summary; the report ensures **every class (0/1/2)** is predicted adequately. fileciteturn7file0

---

## 9) Results (from your notebook runs)

From the printed outputs in your notebook, these were the test accuracies (rounded):

| Model                 | Test Accuracy |
|-----------------------|---------------|
| **Logistic Regression** | **0.8818**     |
| **Random Forest**       | **0.8818**     |
| Gradient Boosting       | 0.8773        |
| Decision Tree           | ~0.8591       |
| KNN (k=5)               | 0.8545        |

> Small differences (±0.5–1%) are common between runs due to randomness; use a fixed `random_state` where possible.

### Plain‑English Conclusions
- **Top performers (tie):** **Logistic Regression** and **Random Forest** (~**0.882**).  
  - Prefer **Logistic Regression** if you want the **simplest** and most **explainable** model.  
  - Prefer **Random Forest** if you want **non‑linear power** and easy **feature importances**.
- **Very close:** **Gradient Boosting** (~0.877). With a **tiny grid search**, it might match or slightly beat RF, but here it’s just behind.  
- **Behind:** **Decision Tree** (single tree often over/underfits) and **KNN** (distance‑based, can struggle if classes overlap).

---

## 10) Hyperparameter Tuning (quick patterns)

Keep tuning **on the pipeline** to avoid leakage.

### 10.1 Gradient Boosting (quick grid)
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gb_base = Pipeline([('prep', preprocess),
                    ('model', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [2, 3]
}

gs_gb = GridSearchCV(gb_base, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs_gb.fit(X_train, y_train)
print("Best params:", gs_gb.best_params_)
evaluate_classifier(y_test, gs_gb.best_estimator_.predict(X_test), "Tuned GradientBoosting")
```

### 10.2 Random Forest (quick grid)
```python
from sklearn.ensemble import RandomForestClassifier

rf_base = Pipeline([('prep', preprocess),
                    ('model', RandomForestClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [200, 300, 500],
    'model__max_depth': [None, 10, 15, 20],
    'model__max_features': ['sqrt', 'log2']
}

gs_rf = GridSearchCV(rf_base, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best params:", gs_rf.best_params_)
evaluate_classifier(y_test, gs_rf.best_estimator_.predict(X_test), "Tuned RandomForest")
```

> If your classes become imbalanced in a future scenario, switch `scoring` to `'f1_macro'` or `'balanced_accuracy'`.

---

## 11) Using **Top‑15 features** to simplify/improve models

The helpers already expose **feature importances** (trees) and **coefficients** (linear) with the **correct post‑preprocessor names**. fileciteturn7file0

### 11.1 Extract Top‑15 from the **winning model**

```python
from functions import feature_importances_dataframe, evaluate_classifier  # helpers
# If Logistic Regression wins:
logr_imp = feature_importances_dataframe(logr_pipeline_fitted, X_fit_columns=X_train.columns)  # coef_* + abs_mean_coef
top15 = logr_imp.head(15)['feature'].tolist()

# If Random Forest wins:
rf_imp  = feature_importances_dataframe(rf_pipeline_fitted, X_fit_columns=X_train.columns)     # importance
top15 = rf_imp.head(15)['feature'].tolist()
```

> For linear multi‑class, the helper uses **abs mean coefficient** across classes to rank features. For trees, it uses **feature_importances_**. fileciteturn7file0

### 11.2 Rebuild a smaller pipeline using only Top‑15

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Map back to raw names if your feature names have prefixes like 'num__'
raw_top15 = [f.split('__')[-1] for f in top15]

preprocess_15 = ColumnTransformer([('num', StandardScaler(), raw_top15)], remainder='drop')

logr_top15 = Pipeline([('prep', preprocess_15),
                       ('model', LogisticRegression(max_iter=1000, random_state=42))])

logr_top15.fit(X_train, y_train)
evaluate_classifier(y_test, logr_top15.predict(X_test), "LogReg (Top‑15 features)")
```

**Why this helps**  
- Models get **simpler & faster** (fewer features).  
- You reduce **noise** and the risk of **overfitting**.  
- Your story becomes clearer: *“These 15 features matter most for predicting stress.”*  
- If accuracy stays roughly the **same**, you’ve gained a **leaner** model with **better interpretability**.

> You can do the same with **Random Forest** or **Gradient Boosting** by swapping the estimator while keeping `preprocess_15`.

---

## 12) Final Takeaways

- With numeric‑only features and balanced classes, your **clean pipeline** + **baseline models** give a trustworthy comparison.  
- In your runs, **Logistic Regression** and **Random Forest** were the **top models** (~**0.882** accuracy), with **Gradient Boosting** close behind. Choose the **simplest** model that meets your goal.  
- Use the **Top‑15 features** workflow to make your model **lighter and easier to explain**—great for presenting to non‑technical stakeholders.  
- Keep everything in **pipelines**, and remember: **split once**, **tune with CV**, **evaluate on the untouched test set**, and **save** the final pipeline for reuse.

---

*Helper functions referenced from `functions.py` (evaluation & feature‑importance utilities). fileciteturn7file0
