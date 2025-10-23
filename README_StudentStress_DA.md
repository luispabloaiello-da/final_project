# Student Stress Data Analysis & Preparation for ML

## Overview

This project analyzes a nationwide **Student Stress Monitoring** dataset to understand **underlying causes of stress** and their **impacts** on students. We organize the exploratory analysis (EDA) and preprocessing so the data are **ready for machine‑learning** models in later stages (classification, clustering, and feature‑based insights).

**Dataset facts (from our analysis summary):**
- **Rows:** 1100
- **Features:** 21
- **Missing values:** none detected
- **Duplicate rows:** 0
- **Target distribution (multi‑class `stress_level`):** {0: 373, 1: 358, 2: 369}
These checks confirm a clean, balanced sample size, with a roughly even split across three stress levels. fileciteturn1file0

We grouped the ~20 variables into five factor families provided by the study design (Psychological, Physiological, Environmental, Academic, Social). Our analysis focuses on **feature relevance** (correlation & mutual information), **outliers**, and **dimensionality reduction** so the dataset is ready for modeling.

---

## Data Sources

- **Student Stress Monitoring dataset** (survey‑based; ~20 features across 5 factor groups).
- **`_main_.ipynb`** — primary analysis notebook used in this project.
- **`analysis_summary.txt`** — machine‑readable summary produced from `_main_.ipynb` runs; used to collect and report the key statistics and recommendations. fileciteturn1file0

---

## Main Files and Structure

### 1. `_main_.ipynb` (primary analysis)

Below we document each analysis step exactly as performed, with **purpose**, **what the code does**, **why it matters for ML**, and a short **insight snippet** you can paste back into the notebook.

#### A) Data shape, schema, and basic integrity
- **Purpose:** Confirm dataset size, feature count, types, and basic quality before any modeling.
- **What the code does:** Loads the CSV into a pandas DataFrame; prints `.shape`, `.dtypes`, checks missing values with `.isna().sum()`, and duplicates via `df.duplicated().sum()`.
- **Why it matters for ML:** Ensures you won’t feed models corrupted or incomplete data; determines the next cleaning steps (none were required here).
- **Insight (paste into notebook):**
  > **Insight:** Dataset has 1100 rows, 21 features, no missing values, and 0 duplicates — ready for analysis. fileciteturn1file0

#### B) Target distribution (`stress_level`)
- **Purpose:** Verify class balance across three levels of stress to choose the right metrics and CV strategy.
- **What the code does:** Uses `value_counts()` and normalized percentages for the target.
- **Why it matters for ML:** Guides metric choice (macro‑F1 vs accuracy), stratified splits, and whether imbalance methods are needed.
- **Insight (paste):**
  > **Insight:** Target is roughly balanced across classes (0: 373, 1: 358, 2: 369), so standard stratified splits and macro‑averaged metrics are appropriate. fileciteturn1file0

#### 8) Descriptive distributions & outliers (numeric features)
- **Purpose:** Inspect spread and unusual values that may bias models.
- **What the code does:** Plots histograms/boxplots; flags outliers using **IQR** (Q1 − 1.5×IQR, Q3 + 1.5×IQR) or z‑scores; counts % of outliers per feature.
- **Why it matters for ML:** Outliers can distort distance‑based models and inflate variance; informs whether to cap/winsorize or use robust models.
- **Insight (paste):**
  > **Insight:** Three features show > 5% outliers; consider winsorization or robust models/metrics if they degrade performance. fileciteturn1file0

#### D) Correlation analysis (target & inter‑feature)
- **Purpose:** Find linear associations with `stress_level` and detect redundant features.
- **What the code does:** Computes Pearson correlations and visualizes a heatmap.
- **Why it matters for ML:** Helps early **feature selection** and alerts to multicollinearity for linear models.
- **Insight (paste):**
  > **Insight:** Top correlates with stress include **bullying**, **future_career_concerns**, **anxiety_level**, **depression**, and **headache**. fileciteturn1file0

#### E) Mutual Information (feature relevance beyond linearity)
- **Purpose:** Capture **non‑linear** relationships between features and `stress_level`.
- **What the code does:** Uses `mutual_info_classif` to score features; sorts descending.
- **Why it matters for ML:** Complements correlation; surfaces predictors valuable to non‑linear models.
- **Insight (paste):**
  > **Insight:** Top MI features: **blood_pressure**, **sleep_quality**, **future_career_concerns**, **depression**, **anxiety_level** — these carry the most information about stress levels. fileciteturn1file0

#### F) Dimensionality reduction (PCA for variance explanation)
- **Purpose:** Understand the **intrinsic dimensionality** and compress features if needed.
- **What the code does:** Standardizes numerics; fits PCA; computes components required to explain 95% of variance.
- **Why it matters for ML:** Guides whether to reduce dimensionality to cut noise and speed training.
- **Insight (paste):**
  > **Insight:** About **16 components** explain **95%** of the variance; PCA can be used as an optional step for compact representations. fileciteturn1file0

#### G) Preprocessing recommendations for modeling
- **Purpose:** Convert analysis findings into a concrete ML‑ready pipeline.
- **What the code does:** Summarizes scaling choice, feature selection breadth, and model family hints.
- **Why it matters for ML:** Ensures reproducible, leak‑free preprocessing and a reasonable starting feature set.
- **Insight (paste):**
  > **Insight:** Use **StandardScaler/MinMaxScaler** for numerics; start with **top ~20 features**; check multicollinearity for linear models; try **ensemble models** (robust baselines). fileciteturn1file0

> **Note:** The notebook also includes standard visual EDA steps (histograms/boxplots/heatmaps) and simple statistics to keep the workflow transparent and student‑friendly (9‑week track level).

---

### 2. `analysis_summary.txt`

A distilled report with key counts, rankings, PCA summary, and preprocessing guidance used above. Keep it updated whenever `_main_.ipynb` changes. fileciteturn1file0

### 3. `README_StudentStress_DA.md` (this file)

Human‑readable documentation of the analysis and how it prepares the dataset for ML.

---

## Key Features of This Project

- **Clean data foundation:** verified shape, types, and integrity (no missing or duplicates). fileciteturn1file0
- **Balanced target:** suitable for stratified CV and macro metrics. fileciteturn1file0
- **Feature relevance from two angles:** **correlation** (linear) and **mutual information** (non‑linear). fileciteturn1file0
- **Outlier awareness:** flags >5% outliers in 3 features with suggested treatments. fileciteturn1file0
- **Dimensionality insight:** PCA shows 95% variance at 16 components — optional compression lever. fileciteturn1file0
- **Actionable preprocessing plan:** scaling, selection, and model family suggestions. fileciteturn1file0

---

## Example Insights (to present in class)

- **Psychological drivers dominate:** anxiety and depression are consistently high across correlation/MI — unsurprising but quantitatively confirmed. fileciteturn1file0
- **Academic pressure matters:** future career concerns strongly co‑move with stress levels, suggesting interventions (career guidance, mentoring). fileciteturn1file0
- **Physiological signals are informative:** blood pressure and sleep quality rank high in MI, indicating measurable health links. fileciteturn1file0
- **Social environment counts:** bullying is among the top correlates, emphasizing the need for anti‑bullying programs. fileciteturn1file0

> **Caution:** Correlation/MI indicate association, **not causation**. Use them for prioritization and modeling, not for causal claims.

---

## How to Use

1. Open **`_main_.ipynb`** and run cells top‑to‑bottom to reproduce the EDA and summaries.
2. Review **`analysis_summary.txt`** for quick stats and copy‑ready tables. fileciteturn1file0
3. Move to modeling notebooks:
   - Start with **Logistic Regression / Random Forest / Gradient Boosting** as baselines.
   - Use **StandardScaler/MinMaxScaler** for numerics; keep categorical encoding if applicable.
   - For linear models, check **collinearity**; for trees, you can keep all dummies.
4. Evaluate with macro‑F1/accuracy (balanced target) and confusion matrices; iterate with feature subsets or PCA as needed.

---

## Files Included

- `_main_.ipynb` — main analysis notebook (EDA & preparation for ML).
- `analysis_summary.txt` — generated summary of key stats and recommendations. fileciteturn1file0
- `README_StudentStress_DA.md` — this documentation file.

---

## Authors

- Luis Pablo Aiello (Student, Data Analytics cohort)  
- Class collaborators (add names as appropriate)

---

## License

For educational use within the bootcamp course. Data derived from a nationwide, survey‑based student stress study.
