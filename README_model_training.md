# Student Stress — Model Training (README)

This README explains the **machine learning models** and **training code** used in `_main_model_training.ipynb`, in the same student-friendly style you've used all course long. It covers **what each model is**, **why we chose it**, **how the code works**, and **how it connects to the lessons (7.1–7.5)**.

> Target variable assumed in the notebook: **`stress_level`** (multi-class: 0, 1, 2).  
> Evaluation helper used: `evaluate_classifier` (prints accuracy, confusion matrix, and classification report). fileciteturn3file0

---

## 0) Imports & Setup (Code cell 1)

**What it does**  
Loads core libraries and scikit-learn utilities used throughout the notebook.

**Key imports observed**  
```python
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
# (and other model-specific imports later in the notebook)
```
- `pandas`, `numpy` → data wrangling and numerics.
- `matplotlib`, `seaborn` → quick plots.
- `train_test_split` → split once and keep test untouched.
- `ColumnTransformer`, scalers, encoders → **preprocessing inside a Pipeline** (prevents leakage).
- `Pipeline` → bundle preprocessing + model so the **same steps** apply to train & test consistently.
- `yaml` (optional) → if the project uses a config file.

**Why this matters**  
This mirrors **7.2 Feature Engineering**: all preprocessing steps are explicit and wrapped in a **scikit-learn Pipeline** to ensure **reproducibility** and no data leakage.

---

## 1) Train/Test split & Preprocessing (Code cell 7)

**What it does**  
Defines the target and splits the data with stratification, then prepares a `ColumnTransformer` + `Pipeline` that scales numeric features and one-hot encodes categoricals.

**Observed pattern**  
```python
target = 'stress_level'
X = df.drop(columns=[target])
y = df[target]

# Identify column types, e.g. numeric vs categorical (not all lines shown in export)
# Build a ColumnTransformer:
#   - ('num', StandardScaler(), num_cols)
#   - ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
# and then wrap it in Pipeline together with your model.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **Split once**: `stratify=y` keeps class proportions similar in train and test.
- **Preprocessing**: consistent with **7.2** lessons—scale numerics, one-hot encode categoricals.

**Why this matters**  
Scaling is important for **distance-based models** (like **KNN**) and helps linear models converge. Encoding converts categoricals to numbers so all models can learn from them.

---

## 2) K-Nearest Neighbors (KNN) Baseline (Code cell 10)

**What is KNN (7.1 lesson)?**  
- A **distance-based** classifier: to predict a new student’s stress level, it looks at the **k closest** students in the training set and **votes**.
- **Key hyperparameter**: `n_neighbors` (k). Small k → flexible (can overfit). Larger k → smoother (can underfit).
- **Needs scaling**: distances are affected by units; use **StandardScaler** in your pipeline.

**Code (from the notebook)**  
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
pipe = Pipeline([('prep', preprocess), ('model', knn)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
evaluate_classifier(y_test, y_pred, title='KNN (k=5)')   # prints accuracy, confusion matrix, and classification report
```

**Line-by-line**  
- `KNeighborsClassifier(n_neighbors=5)` → set k=5 (a sensible starting point).  
- `Pipeline([('prep', preprocess), ('model', knn)])` → ensure your preprocessing is applied before KNN.  
- `.fit(X_train, y_train)` → learn the training set (KNN just stores it).  
- `.predict(X_test)` → for each test row, find 5 nearest neighbors and vote.  
- `evaluate_classifier(...)` → prints metrics. (This helper is defined in `functions.py`.) fileciteturn3file0

**Try a few k values**  
The notebook also loops over k ∈ {3, 5, 7, 9} and prints accuracies, which is a quick bias–variance sanity check (consistent with **7.1**).

---

## 3) Simple Feature Pre-selection + Saving a Pipeline (Code cell 12)

**What is this?**  
A minimal example using **VarianceThreshold** to drop any features with zero variance (constants), and then saving a fitted pipeline to disk with `pickle` (consistent with **7.2_2**).

**Code**  
```python
from sklearn.feature_selection import VarianceThreshold
import pickle

vt = VarianceThreshold(threshold=0.0)
vt_pipe = Pipeline([('prep', preprocess), ('vt', vt)])
Xt = vt_pipe.fit_transform(X_train)
print('Transformed shape:', Xt.shape)

clf = Pipeline([('prep', preprocess), ('model', KNeighborsClassifier(n_neighbors=5))])
clf.fit(X_train, y_train)
with open('KNeighborsClassifier.pkl','wb') as file:
    pickle.dump(clf, file)
print('Saved KNeighborsClassifier.pkl')
```

**Line-by-line**  
- `VarianceThreshold(0.0)` → drop columns with **no variance** (no information).  
- `vt_pipe.fit_transform(X_train)` → fit preprocessing + VT on train and transform it; `Xt.shape` confirms the new dimensionality.  
- Build a **full pipeline** (`prep` + `KNN`) and **fit** it.  
- Save the trained pipeline to `KNeighborsClassifier.pkl` so you can reuse it later (load and `.predict` with the same preprocessing).

**Why this matters**  
This keeps your **preprocessing + model together** so deployment and reproducibility are easy. And you remove obviously useless features safely.

---

## 4) Core Supervised Models (Code cell 15)

This cell compares **Logistic Regression**, **KNN**, and a **Decision Tree** using the **same preprocessing**. This matches **7.3.1 — Supervised ML (core models & metrics)**.

**Code**  
```python
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN(k=5)': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}
for name, est in models.items():
    pipe = Pipeline([('prep', preprocess), ('model', est)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    evaluate_classifier(y_test, y_pred, title=name)     # prints metrics
```

**Line-by-line**  
- `models = {...}` → a dictionary of simple, strong **baselines**.  
- `for name, est in models.items():` → iterate through each algorithm with the **same pipeline**.  
- `Pipeline([('prep', preprocess), ('model', est)])` → identical preprocessing for fair comparison.  
- `fit → predict → evaluate_classifier` → report accuracy, confusion matrix, and class-wise precision/recall/F1.  
  The helper `evaluate_classifier` is defined in `functions.py`. fileciteturn3file0

**What each model is**  
- **Logistic Regression**: linear classifier that outputs class probabilities (interpretable, fast baseline).  
- **KNN**: non-parametric, distance-based, benefits from scaling; good sanity check baseline.  
- **Decision Tree**: learns **if/else** rules; captures non-linear relationships; can overfit if not pruned—use as a simple tree baseline before ensembles.

**Why these were chosen**  
They are **classic, beginner-friendly baselines** covering **linear**, **distance-based**, and **tree-based** families. You can learn different behaviors (bias–variance, interpretability, non-linearity) with very little code and compare them fairly.

---

## 5) Metrics & How to Read Them

The notebook uses `evaluate_classifier(...)` to print:

- **Accuracy** — simple fraction correct (fine here if classes are close to balanced; otherwise prefer macro metrics).  
- **Confusion Matrix** — counts of correct/incorrect by class (helps see which stress levels are confused).  
- **Classification Report** — per-class **precision**, **recall**, **F1**, plus macro/weighted averages.

**Where it comes from**  
`evaluate_classifier` calls scikit-learn’s metrics internally: `accuracy_score`, `confusion_matrix`, `classification_report`. fileciteturn3file0

**Tip**  
If one class is rarer (e.g., stress level 2 is small), focus on **macro-averaged F1** and the **per-class recall** for that minority class (lesson 7.5).

---

## 6) How This Maps to the Lessons

- **7.1 KNN** → you tried KNN with scaling and a few k values.  
- **7.2 Feature Engineering** → preprocessing with **ColumnTransformer** + **Pipeline**; **VarianceThreshold** and saving encoders/models.  
- **7.3.1 Core Models** → compared **LogReg**, **KNN**, **DecisionTree** with consistent metrics.  
- **(Optionally) 7.3.2 Ensembles** → after this baseline, you could add **Random Forest** / **Gradient Boosting** in the same loop.  
- **7.4 Tuning** → once you pick a favorite (e.g., GB), tune `n_estimators`, `max_depth`, `learning_rate` via GridSearch or Optuna.  
- **7.5 Imbalanced** → if class proportions differ a lot, wrap resampling (ROS/RUS/SMOTE) into an **imblearn Pipeline**.

---

## 7) Re-running & Extending

1. **Re-run**: execute the notebook top-to-bottom; note the printed metrics per model.  
2. **Pick a winner**: choose the simplest model that’s accurate enough for your goal (often **LogReg** or a small **DecisionTree** for interpretability; if you add ensembles, **Gradient Boosting** tends to win).  
3. **Tune** (optional): try small grids for the best model.  
4. **Explain**: show confusion matrix and the macro metrics; add a short note: *“Model X is best because it achieved the highest macro F1 while keeping errors balanced across classes.”*

---

## Appendix — Helper functions

- **`evaluate_classifier`**: prints accuracy, confusion matrix, and classification report. This is defined in `functions.py` and imported in your notebook. fileciteturn3file0

---

## 8) **Add‑On: Ensembles (7.3.2) — Random Forest & Gradient Boosting**

**Why ensembles here?**  
After simple baselines (LogReg, KNN, Decision Tree), ensembles often improve performance by combining many weak learners.  
- **Random Forest (bagging):** many trees trained on bootstrap samples + random feature subsets → averaged votes (robust, low tuning).  
- **Gradient Boosting (boosting):** trees are added **sequentially**, each fixing errors of the previous ones (often strongest, needs small/clean tuning).

**Code (drop‑in cell):**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

rf = Pipeline([('prep', preprocess),
               ('model', RandomForestClassifier(n_estimators=300, random_state=42))])

gb = Pipeline([('prep', preprocess),
               ('model', GradientBoostingClassifier(random_state=42))])

for name, pipe in [('RandomForest', rf), ('GradientBoosting', gb)]:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    evaluate_classifier(y_test, y_pred, title=name)
```

**Line‑by‑line highlights**
- `RandomForestClassifier(n_estimators=300)` → more trees usually stabilise results. Key knobs: `n_estimators`, `max_depth`, `max_features`.
- `GradientBoostingClassifier(...)` → strong default; key knobs: `n_estimators`, `learning_rate`, `max_depth`.
- Same **preprocessing** via `('prep', preprocess)` keeps the comparison fair.
- `evaluate_classifier(...)` prints accuracy, confusion matrix, and per‑class precision/recall/F1.

> **Tip:** If you add only one ensemble, start with **Gradient Boosting** (it was best in your previous lab).

---

## 9) **Small Grid Search (7_4) — Tune the Winning Model**

**Goal:** take your current best model (usually **Gradient Boosting**) and try a **small**, safe grid. We keep it simple and fast.

**Code (drop‑in cell):**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

base = Pipeline([('prep', preprocess),
                 ('model', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [2, 3]
}

gs = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',     # for roughly balanced classes; use 'f1_macro' if you want macro F1
    n_jobs=-1,
    verbose=0
)

gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("CV best score:", round(gs.best_score_, 4))

best = gs.best_estimator_
y_pred_best = best.predict(X_test)
evaluate_classifier(y_test, y_pred_best, title='Tuned GradientBoosting (GridSearch)')
```

**Line‑by‑line highlights**
- `base` = your **preprocess + model** pipeline, so CV runs **leak‑free**.
- `param_grid` = only **3 knobs** that matter most for GB (small ranges keep it quick).
- `scoring='accuracy'` is fine if classes are close in size. For imbalance, consider `'f1_macro'` or `'balanced_accuracy'`.
- `best = gs.best_estimator_` returns the **full pipeline** with the best params.
- `evaluate_classifier(...)` gives you final **test** metrics (leave the test set untouched until this step).

> **Optuna option:** If you have `optuna` installed, you can write an objective and try ~25 trials. For a student project, this compact `GridSearchCV` is perfectly acceptable.

---

## 10) **Imbalanced Data (7.5) — ROS / RUS / SMOTE with `imblearn.pipeline.Pipeline`**

If any class becomes **rare** (e.g., <10%), accuracy can be misleading. In that case:
- Use **macro metrics** (macro F1, balanced accuracy).
- Consider **resampling** *on the training set only*. The easiest safe way is to place samplers inside an **`imblearn` Pipeline**.

**⚠️ Important:** Samplers (RandomOverSampler, RandomUnderSampler, SMOTE) must be placed in **`imblearn.pipeline.Pipeline`** (not `sklearn.pipeline.Pipeline`).

**Code (drop‑in cell):**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Baseline (no resampling): can use sklearn's Pipeline
base = Pipeline([('prep', preprocess),
                 ('model', LogisticRegression(max_iter=1000, random_state=42))])

# Resampling variants: MUST use imblearn's Pipeline
ros = ImbPipeline([('prep', preprocess),
                   ('ros', RandomOverSampler(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, random_state=42))])

rus = ImbPipeline([('prep', preprocess),
                   ('rus', RandomUnderSampler(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, random_state=42))])

smt = ImbPipeline([('prep', preprocess),
                   ('smt', SMOTE(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, random_state=42))])

def eval_simple(pipe, title):
    pipe.fit(X_train, y_train)      # resampling happens only on TRAIN
    yp = pipe.predict(X_test)       # test is untouched (as it should be)
    evaluate_classifier(y_test, yp, title)

for title, pipe in [
    ('Baseline LogReg', base),
    ('LogReg + ROS', ros),
    ('LogReg + RUS', rus),
    ('LogReg + SMOTE', smt)
]:
    eval_simple(pipe, title)
```

**Line‑by‑line highlights**
- `ImbPipeline` is required whenever you include a **sampler** step.
- Resampling sits **between** `('prep', preprocess)` and `('model', ...)` so it operates on already‑encoded features and **only on the training fold**.
- Evaluate on the **original test set** (never resample your test!).
- Prefer **macro metrics** and check per‑class **recall** for the minority class.

**Common pitfalls**
- **Error:** “All intermediate steps should be transformers…” → you used **sklearn** `Pipeline` with a sampler. **Fix:** use `imblearn.pipeline.Pipeline`.
- Using SMOTE with **sparse** data (from OneHotEncoder) may fail. Two fixes:
  1) Force dense OHE: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` (or `sparse=False` on older sklearn), or  
  2) Use `SMOTENC` and pass categorical indices.

---

## 11) Presenting Your Results

- **Start with baselines** (LogReg, KNN, Decision Tree) → one table of metrics.  
- Add **ensembles** (RF, GB) → usually stronger; pick the simplest that meets your goal.  
- If needed, show **Grid Search** improvements for the winner.  
- If classes are imbalanced, explain why you used **macro metrics** and (optionally) **resampling**.  
- Include **confusion matrices** and a short narrative:  
  *“Gradient Boosting was best by macro F1; after a small grid search, accuracy improved from 0.78 → 0.80 while keeping class‑2 recall acceptable.”*
