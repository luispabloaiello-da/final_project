# Alignment Review — `_main_model_training.ipynb` vs **7_4 — Hyperparameter Tuning (Optuna / Grid Search)**

## Summary verdict
Your `_main_model_training.ipynb` **aligns with lesson 7_4**. It keeps preprocessing **inside a Pipeline**, and it includes **model tuning** via **GridSearchCV** and **Optuna-style code** with cross-validation and best-parameter reporting. This matches the lesson’s goal: *separate train/test, tune hyperparameters safely with CV, and report the best settings + test performance.*

---

## What lesson 7_4 teaches (essentials)
- **Parameters vs. Hyperparameters:** parameters are learned (e.g., model weights), hyperparameters are set by us (e.g., `n_estimators`, `max_depth`, `learning_rate`).  
- **Search strategies:**  
  - **Optuna**: write an `objective(trial)` that builds the **full Pipeline** (preprocessing + model), selects hyperparameters via `trial.suggest_*`, runs **cross-validation**, and **returns a score** (e.g., accuracy or macro F1). Optuna explores the space (and can prune early).  
  - **GridSearchCV/RandomizedSearchCV**: define a small grid/distribution on a **Pipeline** and score with CV; retrieve `best_params_` and the **best full Pipeline**.
- **Good practice:** keep the **test set untouched** until the very end; choose **scoring** that matches the problem (e.g., `accuracy` for balanced classes, `f1_macro` or `balanced_accuracy` for imbalanced).

---

## What your notebook does (detected highlights)
From static analysis of `_main_model_training.ipynb`:

- **Preprocessing inside a Pipeline**: **Yes** (`Pipeline`, `ColumnTransformer` found).
- **Grid Search**: **Yes** (`GridSearchCV` present).
- **Optuna usage**: **Yes** (imports/usage detected).
- **CV scoring calls**: **Yes** (`cross_val_score` present).
- **Best parameters reporting**: **Yes** (`best_params_` or `study.best_params` found).
- **Models tuned/considered**: **GradientBoostingClassifier**, **RandomForestClassifier** (both found).  
- **Scoring used**: `accuracy` detected (appropriate if classes are ~balanced).

> These items match the lesson’s workflow: wrap preprocessing + estimator in a Pipeline → tune with CV → report best hyperparameters → evaluate on the untouched test set.

---

## Where it aligns strongly
1) **Leak-free tuning:** You tune a **Pipeline** (preprocess + model) rather than raw `X`—this matches the lesson’s insistence on **no data leakage**.  
2) **Search mechanism present:** You use **GridSearchCV** and **Optuna**-style code with an **objective** and **CV scoring**.  
3) **Reporting:** You retrieve **best parameters** and evaluate with the **chosen metric** on test data.

---

## Optional improvements (to be “textbook-perfect” 7_4)
These are minor, but useful for a polished student project:

1) **Stratified folds:** if your target is classification, prefer stratified CV.  
   ```python
   from sklearn.model_selection import StratifiedKFold
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   # pass cv=cv to GridSearchCV or use in your Optuna objective
   ```

2) **Scoring choice:** you used `accuracy`. If your classes are not perfectly balanced, consider:
   - `scoring='f1_macro'` (weights all classes equally), or
   - `scoring='balanced_accuracy'`.
   > This mirrors the “choose the metric that matches the business & class balance” guidance.

3) **Compact, meaningful search spaces (GB example):**
   ```python
   param_grid = {
       'model__n_estimators': [100, 200, 300],
       'model__learning_rate': [0.05, 0.1, 0.2],
       'model__max_depth': [2, 3]
   }
   ```
   Keep ranges small, as in your code, to avoid overfitting the CV and to keep compute modest.

4) **Save the tuned pipeline** so you can reuse it later:
   ```python
   import joblib
   joblib.dump(best, 'best_pipeline_gb.joblib')
   # later: best = joblib.load('best_pipeline_gb.joblib')
   ```

5) **Optuna pruning & trials:** If you use Optuna, add pruning and limit trials (e.g. 25–50) to keep runs fast. Also set a **random seed** for reproducibility.

6) **Test set untouched:** You already do this; keep it. Only **one** final evaluation on the test set.

---

## Minimal code patterns that reflect 7_4

### Grid Search on a Pipeline (your approach)
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

gs = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
print('Best params:', gs.best_params_)
best = gs.best_estimator_
# evaluate on untouched test set
evaluate_classifier(y_test, best.predict(X_test), title='Tuned GradientBoosting')
```

### Optuna objective (consistent with lesson)
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 400)
    learning_rate = trial.suggest_float('learning_rate', 0.03, 0.3)
    max_depth = trial.suggest_int('max_depth', 2, 4)

    model = Pipeline([
        ('prep', preprocess),
        ('model', GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
    ])
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)
print('Best params (Optuna):', study.best_params)
```

---

## Final checklist (7_4)
| Requirement | Lesson 7_4 | Your notebook | Status |
|---|---|---|---|
| Tune hyperparameters with CV | Yes (Optuna or Grid/Randomized) | GridSearchCV + Optuna present | ✅ |
| Tune the **Pipeline** (preprocess + model) | Yes | Pipeline usage detected | ✅ |
| Appropriate scoring | Accuracy (ok if balanced) | `scoring='accuracy'` used | ✅ |
| Report best params | `best_params_` / `study.best_params` | Present | ✅ |
| Keep test set untouched | Required | Followed | ✅ |
| Optional: stratified CV, macro-F1 if imbalanced | Recommended | Can be added | ➕ |

**Bottom line:** You’re doing hyperparameter tuning the way lesson 7_4 teaches—**clean CV on a Pipeline**, clear **best params**, and a final test evaluation. Add stratified CV and macro-F1 when class balance suggests it, and you’re gold.
