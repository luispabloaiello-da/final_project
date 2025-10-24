# Alignment Review — `_main_model_training.ipynb` vs **7.5 — Imbalanced Data**

## Summary verdict
This review checks whether your `_main_model_training.ipynb` follows the **imbalanced‑data practices** taught in lesson **7.5**. Overall alignment is summarized below and followed by concrete recommendations to be “textbook‑perfect” for 7.5.

---

## What lesson 7.5 teaches (essentials)
- **Check class distribution** early (e.g., `y.value_counts(normalize=True)`), and **name the imbalance**.
- Prefer **imbalance‑aware metrics**: **macro‑F1**, **balanced accuracy**, **PR curves & Average Precision** when positives are rare; still show the **confusion matrix**.
- If needed, try **resampling**: **RandomOverSampler (ROS)**, **RandomUnderSampler (RUS)**, **SMOTE/SMOTENC** — **inside an `imblearn.pipeline.Pipeline`** so resampling only touches the **training folds**.
- Alternatively (or additionally), use **class weights** in models that support them.
- Optionally tune **decision thresholds** using **PR or ROC curves** (don’t rely only on 0.5).
- Keep preprocessing in a **Pipeline + ColumnTransformer**, and use **CV with the right scoring**.

---

## What your notebook contains (auto‑detected flags)

**Pipelines & preprocessing**  
- Pipeline: ✅ | ColumnTransformer: ✅

**Distribution checks**  
- `y.value_counts(...)`: ✅  
- Normalized percentages: ❌

**Imbalance‑aware metrics**  
- Macro‑F1 (`f1_macro`): ❌  
- Balanced accuracy: ❌  
- Average Precision (PR‑AUC proxy): ❌  
- Precision‑Recall curve: ❌  
- ROC‑AUC: ❌  
- Confusion matrix / Classification report: ✅ / ✅

**Resampling & class weighting**  
- RandomOverSampler: ✅  
- RandomUnderSampler: ✅  
- SMOTE/SMOTENC: ✅ / ❌  
- `imblearn.pipeline.Pipeline`: ✅  
- Model `class_weight=` usage: ❌

**Threshold tuning**  
- Uses `predict_proba` or `decision_function`: ❌  
- Explicit threshold selection (e.g., PR‑based): ✅

**CV with proper scoring**  
- `cross_val_score` present: ✅  
- CV scoring set to macro/imbalanced: ❌

> Lesson notebook reference targets:  
> - Resampling steps (ROS/RUS/SMOTE) → **✅/✅/✅** and `imblearn.pipeline.Pipeline` → **❌**.  
> - Macro metrics and PR curve/AP → **F1‑macro ❌**, **balanced accuracy ❌**, **PR curve ❌**, **AP ❌**.

---

## Alignment assessment

- **Preprocessing pipeline**: ✔️ Your notebook uses Pipeline + ColumnTransformer consistently.  
- **Class distribution**: ✔️ Distribution checks found. Great to show normalized frequencies.  
- **Metrics**: ✔️ You report basic metrics (confusion matrix/report). Add macro‑F1 or balanced accuracy; consider PR‑AUC (Average Precision) for rare positives.  
- **Resampling**: ✔️ Resampling present with imblearn Pipeline (good practice).  
- **Class weights**: ℹ️ Consider `class_weight='balanced'` for models that support it (LogReg, Trees, SVM).  
- **Thresholds**: ℹ️ If recall on the minority class matters, consider `predict_proba` + PR curve to pick a better threshold than 0.5.  
- **CV scoring**: ✔️ cross_val_score present. If class imbalance is relevant, prefer `scoring='f1_macro'` or `'balanced_accuracy'`.  

---

## Minimal code blocks to be fully aligned with 7.5

### 1) Show class balance
```python
print(y_train.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))
```

### 2) Baseline vs resampled Logistic Regression (use **imblearn Pipeline**)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Baseline (no resampling)
base = Pipeline([('prep', preprocess),
                 ('model', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))])

# ROS / RUS / SMOTE
ros = ImbPipeline([('prep', preprocess), ('ros', RandomOverSampler(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))])

rus = ImbPipeline([('prep', preprocess), ('rus', RandomUnderSampler(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))])

smt = ImbPipeline([('prep', preprocess), ('smt', SMOTE(random_state=42)),
                   ('model', LogisticRegression(max_iter=1000, multi_class='auto', random_state=42))])
```

### 3) Evaluate with macro metrics
```python
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
for name, pipe in [('Baseline LogReg', base), ('LogReg + ROS', ros), ('LogReg + RUS', rus), ('LogReg + SMOTE', smt)]:
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    print(name)
    print('Balanced Acc:', balanced_accuracy_score(y_test, yp))
    print('Macro F1    :', f1_score(y_test, yp, average='macro'))
    print(classification_report(y_test, yp, digits=4, zero_division=0))
```

### 4) (Optional) Threshold tuning using PR curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
pipe = ros  # e.g., your preferred one
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]  # for binary; for multi-class, tune one-vs-rest
ap = average_precision_score(y_test, proba)
prec, rec, thr = precision_recall_curve(y_test, proba)
# pick threshold by desired recall/precision tradeoff, then binarize: (proba >= thr[k])
```

---

## Final checklist (7.5)

| Requirement | Lesson 7.5 | Your notebook | Status |
|---|---|---|---|
| Check class distribution | `value_counts(normalize=True)` | Yes | ✅ |
| Use macro/imbalance metrics | macro‑F1 / balanced acc / PR‑AUC | Add macro metrics | ➕ |
| Show confusion matrix/report | Confusion + report | Yes | ✅ |
| Resampling inside ImbPipeline | ROS/RUS/SMOTE with imblearn Pipeline | Yes | ✅ |
| Class weights where possible | `class_weight='balanced'` | Optional | ➕ |
| Optional threshold tuning | pick threshold via PR/ROC | Add if needed | ➕ |
| CV scoring set for imbalance | `f1_macro` or `balanced_accuracy` | Consider | ➕ |

**Bottom line:** Make sure you (1) quantify the class balance, (2) report **macro‑aware metrics**, and (3) if you resample, place samplers inside an **`imblearn.pipeline.Pipeline`**. Add optional **threshold tuning** when the business goal favors recall or precision on the minority class.
