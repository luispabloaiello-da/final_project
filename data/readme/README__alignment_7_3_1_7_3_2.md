# Alignment Review — `_main_model_training.ipynb` vs Lessons **7.3.1** & **7.3.2**

## Summary verdict
- **7.3.1 alignment (core classifiers & metrics):** ✅ Your notebook includes the three core models (**Logistic Regression**, **KNN**, **Decision Tree**) and evaluates them with the **right student-friendly metrics** (accuracy, confusion matrix, classification report). Preprocessing is kept inside a **Pipeline** with a **ColumnTransformer**, consistent with earlier lessons.
- **7.3.2 alignment (ensembles):** ✅ If you have the ensemble section in your notebook, it is aligned when you add **Random Forest** and **Gradient Boosting** (and optionally AdaBoost). If that section is currently absent in `_main_model_training.ipynb`, the lessons recommend adding a **small comparison cell** for **RF** and **GB** using the same preprocessing pipeline—see the code block in the “What to add (if missing)” section.

---

## What 7.3.1 expects
- Train **Logistic Regression**, **KNN**, and **Decision Tree** on the **same preprocessed features**.
- Use a **Pipeline + ColumnTransformer** so preprocessing is leak-free and reproducible.
- Report **accuracy**, **confusion matrix**, and **classification report** (precision/recall/F1 per class).
- (Optional) Show **ROC/PR** curves for binary tasks, or use macro-averaged metrics for multi-class.

### What your `_main_model_training.ipynb` does
- **Models present:** LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier.
- **Preprocessing:** `ColumnTransformer` (scaler for numeric + OHE for categoricals) inside `Pipeline`.
- **Metrics:** accuracy, confusion matrix, classification report (macro averages printed by sklearn’s report).
- **Conclusion:** ✔️ **Aligned** with 7.3.1.

**Minor suggestions**
- If you want to reflect the lesson’s discussion on **bias–variance**, you can show a tiny parameter sweep for the tree (e.g., `max_depth`) and for KNN (`n_neighbors`) using cross-validation to illustrate under/overfitting.

---

## What 7.3.2 expects (ensembles)
- Add **Random Forest (bagging)** and **Gradient Boosting (boosting)** as stronger baselines.
- Keep the **same preprocessing** and **evaluation** for fair comparison.
- Briefly mention the main knobs:
  - **RF:** `n_estimators`, `max_depth`, `max_features`.
  - **GB:** `n_estimators`, `learning_rate`, `max_depth` (or `max_leaf_nodes`).

### What your `_main_model_training.ipynb` does
- If your current notebook already includes **RandomForestClassifier** and **GradientBoostingClassifier** with the same pipeline and prints the same metrics → ✔️ **Aligned**.
- If not yet included → add the short block below to align fully.

### What to add (if missing)
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
    evaluate_classifier(y_test, y_pred, title=name)  # accuracy + confusion + per-class precision/recall/F1
```

---

## Metrics: are they handled as in the lessons?
- **Yes.** You use accuracy, confusion matrix, and classification report (precision/recall/F1 per class), which matches 7.3.1 guidance for student projects.  
- If you later switch to a **binary** or **imbalanced** version of the problem, consider **PR-AUC** and **macro-F1** as the primary yardsticks (as seen in 7.5).

---

## Cross-validation & tuning (bridge to 7_4)
- 7.3.1/7.3.2 stop at clean baseline comparisons. For the next step (7_4), you can add a **small GridSearch** for your current winner (often **Gradient Boosting**) using the **same pipeline** so there is no leakage.

**Tiny example**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

base = Pipeline([('prep', preprocess),
                 ('model', GradientBoostingClassifier(random_state=42))])

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [2, 3]
}

gs = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
best = gs.best_estimator_
evaluate_classifier(y_test, best.predict(X_test), title='Tuned GradientBoosting')
```

---

## Final checklist

| Requirement | In lessons | In `_main_model_training.ipynb` | Status |
|---|---|---|---|
| Logistic Regression baseline | 7.3.1 | Yes | ✅ |
| KNN baseline | 7.3.1 | Yes | ✅ |
| Decision Tree baseline | 7.3.1 | Yes | ✅ |
| Pipeline + ColumnTransformer | 7.3.1 | Yes | ✅ |
| Accuracy + confusion + class report | 7.3.1 | Yes | ✅ |
| Random Forest | 7.3.2 | Add if missing | ✅/➕ |
| Gradient Boosting | 7.3.2 | Add if missing | ✅/➕ |
| (Optional) AdaBoost | 7.3.2 (extension) | Optional | ➕ |
| Consistent evaluation across models | 7.3.1/7.3.2 | Yes | ✅ |

**Bottom line:** Your notebook follows the **core model comparison** approach of **7.3.1** and, when you include **RF/GB**, it matches **7.3.2** as well. Keep the preprocessing pipeline constant across models and report the same metrics—that’s exactly what the lessons teach.
