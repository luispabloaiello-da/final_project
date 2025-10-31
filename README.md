# Student Stress Prediction — End-to-End ML Project

> **Links**  
> **Slides:** → [Student Stress Prediction](https://www.canva.com/design/DAG3FBg1QyA/JIeaf5_BqknLrb2mvHibbw/edit?utm_content=DAG3FBg1QyA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)  
> **Streamlit app:** → [Student Stress Prediction— Model Explorer (Dropdown Inputs)](https://finalproject-3ayxr2iyk4pzmubour3zpm.streamlit.app/#student-stress-model-explorer) 

## Overview
This project predicts a student’s **stress level** (multi-class target: `0`, `1`, `2`) from ~20 **numeric survey features** spanning **psychological**, **physiological**, **environmental**, **academic**, and **social** factors.  
It includes **reproducible EDA**, **leak-free ML pipelines**, **model comparisons**, and a **Streamlit app** for interactive scoring and evaluation.  
The goal is a full analytics & ML workflow that’s easy for data-analytics students to follow and extend.

---

## What You’ll Find
- **Clean EDA & Prep:** distributions, outliers, correlations, **multicollinearity** checks (incl. VIF), **Mutual Information**, and a brief **PCA** summary.  
- **Modeling:** consistent pipelines (shared preprocessing), multiple classifiers, optional resampling (**SMOTE**), and feature importance/coefficients helpers.  
- **Comparisons:** accuracy + macro precision/recall/F1, classification reports, **confusion matrices**, and a clear **Top-2 model** selection.  
- **Deployment:** export of fitted pipelines and a **Streamlit** app for single/batch predictions and on-app evaluation (pattern inspired by a prior “fraud scoring” project template).

---

## Data Snapshot
From the EDA:
- **Rows:** ~1,100  
- **Features:** ~21 (all numeric)  
- **Missing values:** none  
- **Duplicates:** none  
- **Target distribution (`stress_level`):** `{0: 373, 1: 358, 2: 369}` → roughly balanced, so **accuracy** is a fair primary metric.

> Note: The dataset is for learning; values reflect typical student self-reports (not clinical labels).

---

## Repo Structure

```
final_project/
├── data/
│ ├── raw/                          # original files
│ └── clean/                        # cleaned/processed artifacts
├── figures/                        # saved figures from EDA
├── notebooks/
│ ├── _main_dataset_analysis.ipynb  # EDA & preparation
│ └── _main_model_training.ipynb    # model training, evaluation, export
├── lib/
│ └── functions.py                  # helpers (run_models_with_importances, importance tables, plotting)
├── my_streamlit_app/
│ ├── app.py                        # Streamlit app (single/batch predict + evaluation)
│ ├── pages/01_Compare_Models.py    # optional: side-by-side model comparison page
│ └── models/
│ ├── *.pkl                         # exported fitted pipelines (incl. tuned “(best)” variants)
│ ├── feature_names.pkl             # full training schema (ALL features)
│ ├── test_set.csv                  # held-out test set for on-app evaluation
│ └── metrics.json                  # quick accuracy per model
├── README_StudentStress_DA_UPDATED.md          # detailed EDA notes & findings
├── README_model_training_results_UPDATED.md    # modeling results & selection details
└── README.md                                   # this file
```

---

## Environment & Installation (uv)

This project targets Python **3.13** (see `requires-python = ">=3.13"`). If your system default is lower, create the venv with an explicit 3.13 interpreter. 

1) **Check you have uv**

   `uv --version`

2) **Create & activate a virtual environment**

- **Windows (PowerShell)**

   `uv venv .venv --python 3.13`
   `.\.venv\Scripts\Activate.ps1`

3) **Install dependencies (choose one profile)**

   A) **Production (app + inference only)**
   - Installs the minimal set to run the Streamlit app and unpickle pipelines:

      `uv pip install -r requirements-dev.txt`

   >  Alternative: if you prefer installing from pyproject.toml, run:
      
   >   `uv sync`

   -  This will install everything declared there (includes Jupyter and ipykernel). Use this only if you want notebooks in the same env.

4) **Optional system dependency**

- **Graphviz** system binary (for certain visualizations) may be required separately from `graphviz` Python bindings depending on your OS.

---

## End-to-End Workflow
1) **EDA & Preparation** (`notebooks/_main_dataset_analysis.ipynb`)  
   - Univariate distributions (skew, mean vs median)  
   - Bivariate feature vs stress (box/violin, group stats)  
   - Correlation (feature–feature & feature–target) + multicollinearity scanning  
   - Outliers (IQR / z-score)  
   - **Mutual Information** (non-linear relevance)  
   - **PCA** for variance explanation (optional dimensionality reduction)  
   _All steps include concise what/why/how notes for teaching clarity._

2) **Modeling** (`notebooks/_main_model_training.ipynb`)  
   - Leak-free pipelines with shared preprocessing evaluated across:  
     **KNN, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost**, and **LogReg + SMOTE**.  
   - Unified evaluation (accuracy, classification report, confusion matrix) and feature importance/coefficients where available.

3) **Export & Deploy**  
   - A final cell saves each **fitted pipeline** (`*.pkl`), the training **feature schema** (`feature_names.pkl`), the **test set** CSV, and a **metrics** JSON.  
   - The **Streamlit** app loads these artifacts for single/batch prediction and on-app evaluation.

---

## Results (short version)
**Test Accuracy (top models):**
- **Random Forest** → best **overall balance** (Acc. 0.8955, Macro F1 0.8953)
- **Logistic Reg + RUS** → Acc. **0.8909**, simplest to explain, good class-0 recall
- **Logistic Reg + SMOTE** → Acc. **0.8909**, best recall on stressed classes (1, 2)
- **Gradient Boosting + hiperparam** → Acc. 0.8955
- Confusion matrices: mostly diagonal, only 1 ↔ 2 mix
- Top features consistent across models


**What the top-features say (recurring signals):**  
High association with stress appeared for: **blood_pressure, sleep_quality, social_support, anxiety_level, depression, self_esteem, bullying, academic_performance, study_load, future_career_concerns**.  
*(Direction depends on feature; e.g., self-esteem often decreases as stress increases.)*

> **Why these two?**  
They showed the best overall balance across classes (macro metrics) with strong recall on stressed classes while maintaining solid precision.

---

## How to Reproduce
**Environment:** Python 3.10+ · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `matplotlib` · `joblib` · `streamlit`

**Steps**
1. **Run EDA**: open `_main_dataset_analysis.ipynb`, execute all cells, check that the snapshot matches the counts above.  
2. **Train models**: open `_main_model_training.ipynb`, run the training & evaluation cells (e.g., `run_models_with_importances(...)`) and review comparisons.  
3. **Export for Streamlit**: run the final **“Streamlit Export Setup (Stress)”** cell → writes `*.pkl`, `feature_names.pkl`, `test_set.csv`, and `metrics.json` into `my_streamlit_app/models/`.  
4. **Launch the app (local)**:
   ```bash
   cd my_streamlit_app
   pip install -r requirements.txt
   streamlit run app.py

> Run the export **after** training so all fitted pipelines (e.g., AdaBoost) are in memory to be saved.

---

## Streamlit App
**Local usage**
```bash
cd my_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## Authors:

- **Luis Pablo Aiello** — Data Analytics Student (Cohort Sep-2025)

---

## License
Educational use within the bootcamp cohort; dataset is survey-based and used for learning purposes.