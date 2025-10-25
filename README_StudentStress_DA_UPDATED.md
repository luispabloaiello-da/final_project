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
- **`_main_dataset_analysis.ipynb`** — primary analysis notebook used in this project.
- **`analysis_summary.txt`** — machine‑readable summary produced from `_main_dataset_analysis.ipynb` runs; used to collect and report the key statistics and recommendations. fileciteturn1file0

---

## Main Files and Structure

### 1. `_main_dataset_analysis.ipynb` (primary analysis)

Below we document each analysis step exactly as performed, with **purpose**, **what the code does**, **why it matters for ML**, and a short **insight snippet** you can paste back into the notebook.

#### 1) Data shape, schema, and basic integrity
- **Purpose:** Confirm dataset size, feature count, types, and basic quality before any modeling.
- **What the code does:** Loads the CSV into a pandas DataFrame; prints `.shape`, `.dtypes`, checks missing values with `.isna().sum()`, and duplicates via `df.duplicated().sum()`.
- **Why it matters for ML:** Ensures you won’t feed models corrupted or incomplete data; determines the next cleaning steps (none were required here).
> **Insight:** Dataset has 1100 rows, 21 features, no missing values, and 0 duplicates — ready for analysis.

#### 2) UNIVARIATE ANALYSIS — DISTRIBUTION OF FEATURES
- **Purpose:** Understand each feature’s shape (center, spread, skew, tails) to inform preprocessing and model choice.
- **What the code does:** Loops over all columns and draws **histograms** with **mean** (green dashed) and **median** (orange dashed) lines; saves the composite figure to config['output_data']['fig1']. Then prints a **skewness** table (sorted).
- **Why it matters for ML:** Clear **skew** (mean ≠ median, high skewness) can distort distance-based models and violate linear assumptions; this guides whether to apply **transformations** (e.g., log/Power for positive variables), **robust scaling**, or prefer **tree/ensemble** models less sensitive to distribution shape.
> **Insight:** Distributions indicate **mild right-skew** in **headache, study_load, depression, noise_level, future_career_concerns, teacher_student_relationship, peer_pressure** and **mild left-skew** in **self_esteem, blood_pressure, social_support**. Skews are generally small, so no transform is strictly required.

#### 3) Target distribution (`stress_level`)
- **Purpose:** Verify class balance across three levels of stress to choose the right metrics and CV strategy.
- **What the code does:** Uses `value_counts()` and normalized percentages for the target.
- **Why it matters for ML:** Guides metric choice (macro‑F1 vs accuracy), stratified splits, and whether imbalance methods are needed.
> **Insight:** Target is roughly balanced across classes (0: 373, 1: 358, 2: 369), so standard stratified splits and macro‑averaged metrics are appropriate.

#### 4) Correlation analysis (target & inter‑feature)
- **Purpose:** Find linear associations with `stress_level` and detect redundant features.
- **What the code does:** Computes Pearson correlations and visualizes a heatmap.
- **Why it matters for ML:** Helps early **feature selection** and alerts to multicollinearity for linear models.
> **Insight:** Top correlates with stress include **bullying**, **future_career_concerns**, **anxiety_level**, **depression**, and **headache**.

#### 5) MULTICOLLINEARITY DETECTION
- **Purpose:** Detect highly correlated predictor pairs that can create redundancy and unstable coefficients in linear/logistic models.
- **What the code does:** Scans the **upper triangle** of the inter-feature **Pearson correlation matrix** and collects pairs with **|r| > 0.7** (the notebook’s threshold). Prints a sorted table of those pairs; otherwise reports “no highly correlated pairs found.”
- **Why it matters for ML:** Strong inter-feature correlation can inflate variance and blur interpretation. If pairs exceed the threshold, either **drop one feature from each pair** (choose by domain value, missingness, or predictive signal) or prefer **regularized** or **tree/ensemble** models that are less sensitive to collinearity.
> **Insight:** For **Multicollinearity Detection** we should ignore correlations with the **target** (`stress_level`) and only act on **feature–feature** pairs.

**High-correlation pairs (|r| > 0.7):**
- `anxiety_level` – `future_career_concerns`: r = 0.717
- `future_career_concerns` – `bullying`: r = 0.711
- `anxiety_level` – `bullying`: r = 0.710
- `depression` – `future_career_concerns`: r = 0.707
- `anxiety_level` – `sleep_quality`: r = −0.710
- `self_esteem` – `future_career_concerns`: r = −0.713
- `blood_pressure` – `social_support`: r = −0.753
- **We drop:** `anxiety_level`, `future_career_concerns`, `social_support`
- **We keep:** `bullying`, `depression`, `sleep_quality`, `self_esteem`, `blood_pressure`
> Removing **anxiety_level** and **future_career_concerns** breaks the dense 0.70+ cluster with **bullying**, **depression**, **sleep_quality**, and **self_esteem**; dropping **social_support** (while keeping **blood_pressure**) resolves the |r|=0.753 pair. This yields a feature set without any remaining |r| > 0.7 **between predictors**. (Correlations with stress_level are expected and were **not** used for dropping; they reflect predictive signal, not multicollinearity.)

#### 6) BIVARIATE ANALYSIS — FEATURE vs STRESS LEVEL
- **Purpose:** Visualize how each feature differs across `stress_level` classes and assess separability.
- **What the code does:** Plots **violin/box** per feature by `stress_level`; computes per‑class **means/medians/std**; highlights largest standardized mean differences (0 vs 2).
- **Why it matters for ML:** Confirms which predictors have clear between‑class shifts, supporting feature prioritization and thresholding choices later.
> **Insight:** Strongest separation across stress levels for **self_esteem**, **bullying**, **sleep_quality**, **future_career_concerns**, **anxiety_level**, **depression**, **academic_performance**. Directionally (from low to high stress): **bullying**, **future_career_concerns**, **anxiety_level**, and **depression** tend to **increase**, while **self_esteem**, **sleep_quality**, and **academic_performance** tend to **decrease**.

**How assessed “separation”**
- Plotted each feature vs `stress_level` (0/1/2) using box/violin charts and printed per-class **means/medians/std**.
- Look for two signals:
    1) **Large, monotonic shifts in the medians** from low → high stress (0 → 2), with **small overlap** between boxes/violins.
    2) Where available, the stats section (ANOVA/Kruskal) supports those visual gaps with **significant p-values** and **non-trivial effect sizes**.

**Why these specific features:**
- **`bullying` — ↑ with stress:** Distributions shift upward as stress increases; high-stress students cluster at higher bullying scores, with the median clearly rising and limited overlap with the low-stress group.
- **`future_career_concerns` — ↑ with stress:** Similar pattern to bullying—consistent upward shift across classes and visibly higher center/spread in stressed groups.
- **`anxiety_level` — ↑ with stress:** Clear upward trend in the central tendency; high-stress group sits notably above low stress.
- **`depression` — ↑ with stress:** Medians increase stepwise; high-stress tails are heavier and overlap is reduced.
- **`self_esteem` — ↓ with stress:** The direction flips; higher stress aligns with lower self-esteem. Boxes/violins are shifted downward with less overlap at the extremes.
- **`sleep_quality` — ↓ with stress:** Another protective factor; medians drop as stress increases, and distributions for high stress are concentrated at worse sleep values.
- **`academic_performance` — ↓ with stress:** Central tendency declines with stress, and the high-stress group shows a lower median and distribution shift.

**What this means for modeling?**
- These features show **class-separating signal** by themselves—useful for both linear and non-linear models.
- Because some of the “↑ with stress” features are correlated with each other (e.g., anxiety/depression/future_career_concerns/bullying), avoid keeping **all** of them unregularized in linear models. 

#### 7) Descriptive distributions & outliers (numeric features)
- **Purpose:** Inspect spread and unusual values that may bias models.
- **What the code does:** Plots histograms/boxplots; flags outliers using **IQR** (Q1 − 1.5×IQR, Q3 + 1.5×IQR) or z‑scores; counts % of outliers per feature.
- **Why it matters for ML:** Outliers can distort distance‑based models and inflate variance; informs whether to cap/winsorize or use robust models.
> **Insight:** Three features show > 5% outliers; consider winsorization or robust models/metrics if they degrade performance.

#### 8) Mutual Information (feature relevance beyond linearity)
- **Purpose:** Capture **non‑linear** relationships between features and `stress_level`.
- **What the code does:** Uses `mutual_info_classif` to score features; sorts descending.
- **Why it matters for ML:** Complements correlation; surfaces predictors valuable to non‑linear models.
> **Insight:** Top MI features: **blood_pressure**, **sleep_quality**, **future_career_concerns**, **depression**, **anxiety_level** — these carry the most information about stress levels.

#### 9) Dimensionality reduction (PCA for variance explanation)
- **Purpose:** Understand the **intrinsic dimensionality** and compress features if needed.
- **What the code does:** Standardizes numerics; fits PCA; computes components required to explain 95% of variance.
- **Why it matters for ML:** Guides whether to reduce dimensionality to cut noise and speed training.
  > **Insight:** About **16 components** explain **95%** of the variance; PCA can be used as an optional step for compact representations.

#### 10) STATISTICAL TESTS
- **Purpose:** Check whether selected features differ in **mean** across stress_level groups.
- **What the code does:** Per feature, checks **normality (Shapiro)** and **variance homogeneity (Levene)** to choose **ANOVA + Tukey** or **Kruskal–Wallis + Dunn (Holm)**; reports **p‑values** and **effect sizes** (η² / ε²).
- **Why it matters for ML:** Validates EDA patterns, ranks features by statistical signal, and informs which effects are **small/medium/large** for model focus.
  > **Insight:** ANOVA shows significant mean differences across stress levels for **`anxiety_level` (F = 655.45, p < 0.001), `self_esteem` (F = 775.39, p < 0.001), `depression` (F = 652.63, p < 0.001), and `academic_performance` (F = 639.22, p < 0.001)**. Largest shifts appear between stress **0 vs 2** (confirm via group means/plots). These features are strong candidates for the model and for monitoring class separation.

#### 11) Preprocessing recommendations for modeling
- **Purpose:** Convert analysis findings into a concrete ML‑ready pipeline.
- **What the code does:** Summarizes scaling choice, feature selection breadth, and model family hints.
- **Why it matters for ML:** Ensures reproducible, leak‑free preprocessing and a reasonable starting feature set.
- **Insight (paste):**
> **Insight:** Use **StandardScaler/MinMaxScaler** for numerics; start with **top ~20 features**; check multicollinearity for linear models; try **ensemble models** (robust baselines).

> **Note:** The notebook also includes standard visual EDA steps (histograms/boxplots/heatmaps) and simple statistics to keep the workflow transparent and student‑friendly (9‑week track level).

---

### 2. `analysis_summary.txt`

A distilled report with key counts, rankings, PCA summary, and preprocessing guidance used above. Keep it updated whenever `_main_.ipynb` changes.

### 3. `README_StudentStress_DA.md` (this file)

Human‑readable documentation of the analysis and how it prepares the dataset for ML.

---

## Key Features of This Project

- **Clean data foundation:** verified shape, types, and integrity (no missing or duplicates). 
- **Balanced target:** suitable for stratified CV and macro metrics. 
- **Feature relevance from two angles:** **correlation** (linear) and **mutual information** (non‑linear). 
- **Outlier awareness:** flags >5% outliers in 3 features with suggested treatments. 
- **Dimensionality insight:** PCA shows 95% variance at 16 components — optional compression lever. 
- **Actionable preprocessing plan:** scaling, selection, and model family suggestions. 

---

## Example Insights (to present in class)

- **Psychological drivers dominate:** anxiety and depression are consistently high across correlation/MI — unsurprising but quantitatively confirmed. 
- **Academic pressure matters:** future career concerns strongly co‑move with stress levels, suggesting interventions (career guidance, mentoring). 
- **Physiological signals are informative:** blood pressure and sleep quality rank high in MI, indicating measurable health links. 
- **Social environment counts:** bullying is among the top correlates, emphasizing the need for anti‑bullying programs. 

> **Caution:** Correlation/MI indicate association, **not causation**. Use them for prioritization and modeling, not for causal claims.

---

## How to Use

1. Open **`_main_dataset_analysis.ipynb`** and run cells top‑to‑bottom to reproduce the EDA and summaries.
2. Review **`analysis_summary.txt`** for quick stats and copy‑ready tables. 
3. Move to modeling notebooks:
   - Start with **Logistic Regression / Random Forest / Gradient Boosting** as baselines.
   - Use **StandardScaler/MinMaxScaler** for numerics; keep categorical encoding if applicable.
   - For linear models, check **collinearity**; for trees, you can keep all dummies.
4. Evaluate with macro‑F1/accuracy (balanced target) and confusion matrices; iterate with feature subsets or PCA as needed.

---

## Files Included

- `_main_dataset_analysis.ipynb` — main analysis notebook (EDA & preparation for ML).
- `analysis_summary.txt` — generated summary of key stats and recommendations. 
- `README_StudentStress_DA.md` — this documentation file.

---

## Authors

- Luis Pablo Aiello (Student, Data Analytics cohort)  

---

## License

For educational use within the bootcamp course. Data derived from a nationwide, survey‑based student stress study.