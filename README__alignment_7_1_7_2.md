# Alignment Review — `_main_model_training.ipynb` vs Lessons **7.1** & **7.2**

**Verdict:** Your `_main_model_training.ipynb` **aligns well** with lessons **7.1** (KNN & supervised intro) and **7.2** (feature engineering with pipelines).  
It follows the course best practices: split once with stratification, keep preprocessing inside a `Pipeline`/`ColumnTransformer`, train a simple KNN baseline, try a few k values, perform minimal pre‑selection, and save a fitted pipeline.

---

## 1) Alignment with **7.1 — Intro to Supervised ML (KNN)**

### What 7.1 expects
- Use a simple **baseline** (KNN).
- **Split once** with `train_test_split(..., stratify=y)` (classification).
- **Scale numeric** features (KNN is distance‑based).
- Keep preprocessing **inside a Pipeline** to avoid leakage.
- Evaluate with **accuracy, confusion matrix, classification report**.
- Try a few **k** values to see bias–variance.

### What your notebook does
- Trains **KNN(k=5)** as a baseline.
- Uses **`train_test_split(..., stratify=y)`**.
- Builds **`ColumnTransformer` + `Pipeline`** with **`StandardScaler`** (numeric) and **`OneHotEncoder(handle_unknown='ignore')`** (categorical).
- Prints **accuracy + confusion matrix + per‑class precision/recall/F1** (via `evaluate_classifier`).
- Tries **k ∈ {3,5,7,9}** for a quick bias–variance check.

**Status:** ✅ **Fully aligned with 7.1.**

**Tiny improvements (optional)**
```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

for k in [3,5,7,9,11]:
    m = Pipeline([('prep', preprocess),
                  ('model', KNeighborsClassifier(n_neighbors=k))])
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f'k={k}: CV accuracy = {cv:.4f}')
```
- Consider `weights='distance'` for overlapping classes.

---

## 2) Alignment with **7.2_1 — Feature Engineering (Pipelines & Preprocessing)**

### What 7.2_1 expects
- Preprocessing **inside** a `Pipeline` / `ColumnTransformer`:
  - **Scale** numeric features (e.g., `StandardScaler`).
  - **One‑hot encode** categoricals (`OneHotEncoder(handle_unknown='ignore')`).
- Avoid **data leakage**: fit transforms **only** on the training data.

### What your notebook does
- Uses a **`ColumnTransformer`**:
  - `('num', StandardScaler(), num_cols)`
  - `('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)`
- Wraps it in a **`Pipeline`** with each estimator.
- Trains/evaluates through the same pipeline ⇒ **leak‑free** preprocessing.

**Status:** ✅ **Fully aligned with 7.2_1.**

**Tiny improvements (optional)**
- If you have **binary‑like** numeric flags, letting them **pass through** unscaled is fine (minor).
- If a later step can’t handle sparse matrices, set `OneHotEncoder(..., sparse_output=False)` (or `sparse=False` in older sklearn).

---

## 3) Alignment with **7.2_2 — Feature Engineering (Pre‑selecting & Saving Encoders)**

### What 7.2_2 expects
- Demonstrate a **simple, safe pre‑selection** (e.g., `VarianceThreshold`) to remove constant columns.
- **Persist** the fitted preprocessing + model (pickle/joblib) so inference uses identical transforms.

### What your notebook does
- Uses **`VarianceThreshold(threshold=0.0)`** to drop zero‑variance features.
- Saves a fitted **pipeline** (preprocess + model) to disk with `pickle.dump(...)`.

**Status:** ✅ **Fully aligned with 7.2_2.**

**Tiny improvements (optional)**
- Prefer **`joblib.dump`** for large objects (faster/safer for NumPy arrays).
- Record the **scikit‑learn version** when saving artifacts to avoid compatibility issues.

---

## 4) Quick Pass/Fail Matrix

| Topic | Expectation (lessons) | Found in your notebook | Status |
|---|---|---|---|
| KNN baseline | KNN with scaled features & split | KNN(k=5) + StandardScaler + stratified split | ✅ |
| Pipeline & CT | `ColumnTransformer` + `Pipeline` (no leakage) | Preprocess embedded in pipelines for all models | ✅ |
| Metrics | Accuracy, confusion matrix, class report | Printed via `evaluate_classifier` | ✅ |
| Try k values | Quick sweep (bias–variance) | Tried 3/5/7/9 | ✅ |
| Pre‑selection | Simple & safe | `VarianceThreshold(0.0)` | ✅ |
| Save pipeline | Persist preprocess+model | `pickle.dump(...)` on pipeline | ✅ |

---

## 5) Recommendations (Optional polish)
1) Add a tiny **CV** cell to pick `k` for KNN systematically (snippet above).
2) If you later add **resampling** (ROS/RUS/SMOTE), remember: sampler steps must be inside **`imblearn.pipeline.Pipeline`**, not sklearn’s pipeline.
3) When saving artifacts for production, switch to **`joblib.dump`** and log library versions.

---

### Bottom line
Your `_main_model_training.ipynb` follows the **exact methods from 7.1 & 7.2**: clean split, pipeline‑based preprocessing (scaling + OHE), a clear KNN baseline with a small k‑sweep, minimal feature pre‑selection, and pipeline persistence.

