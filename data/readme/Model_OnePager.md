# Student Stress — Model One‑Pager

- Target: multi‑class (0/1/2); Features: numeric‑only; Balance: not imbalanced
- Preprocessing: StandardScaler inside a Pipeline (leak‑free)

## Models (test accuracy)

| Model             |   Accuracy |
|:------------------|-----------:|
| AdaBoost          |     0.8864 |
| RandomForest      |     0.8864 |
| LogReg + SMOTE    |     0.8864 |
| Random Forest     |     0.8818 |
| Baseline LogReg   |     0.8818 |
| LogReg + ROS      |     0.8818 |
| LogReg + RUS      |     0.8818 |
| Gradient Boosting |     0.8773 |
| DecisionTree      |     0.8591 |
| KNN (k=5)         |     0.8545 |

## Conclusions
- Accuracy is a fair primary metric here; confirm per‑class report.
- Logistic Regression = simple & interpretable. Random Forest = non‑linear + importances.
- Use Top‑15 features for a lighter model with similar accuracy.