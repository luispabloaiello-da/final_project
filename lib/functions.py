#================================================================
# Add all the imports needed by the functions in the project here
#================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re

from scipy import stats
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

try:
    # If you also use imblearn Pipelines for resampling (ROS/RUS/SMOTE)
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    ImbPipeline = tuple()  # harmless fallback if imblearn isn't present


#================================================================
# Custom preprocessing functions
#================================================================

# Exclude "binary-like" numerics (e.g., 0/1 flags)
def is_binary_like(series: pd.Series) -> bool:
    s = series.dropna()
    # Treat as binary-like if only 2 unique values OR subset of {0,1}
    uniq = pd.unique(s.values)
    if len(uniq) <= 2:
        return True
    try:
        return set(np.unique(s)).issubset({0, 1})
    except Exception:
        return False

# Return a compact, readable p-value string
def _pretty_p(p: float, threshold: float = 1e-4) -> str:
    if np.isnan(p):
        return "p = NaN"
    return f"< {threshold:.4f}" if p < threshold else f"{p:.4f}"

#================================================================
# Custom preprocessing functions for visualizations
#================================================================

# Simple IQR Limit Calculation
def iqr_bounds(s: pd.Series, k: float = 1.5):
    x = s.dropna().values
    if len(x) == 0:
        return np.nan, np.nan
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

def quick_outlier_scan(df: pd.DataFrame, cols=[], k: float = 1.5, title_prefix:str=""):
    for c in cols:
        lo, hi = iqr_bounds(df[c], k)
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        print(f"{title_prefix}{c}: outliers={n_out}, low={lo:.3f}, high={hi:.3f}")

        # Boxplot with IQR limits
        ax = sns.boxplot(x=df[c])
        if np.isfinite(lo): ax.axvline(lo, ls='--')
        if np.isfinite(hi): ax.axvline(hi, ls='--')
        ax.set_title(f"{title_prefix}{c} (box + IQR)"); 
        plt.show()

def drop_outliers_iqr(df: pd.DataFrame, cols=[], k: float = 1.5, how:str='any'):
    """
    how='any' -> deletes a row if it's an outlier in at least one column
    how='all' -> deletes a row only if it's an outlier in all columns
    """
    d = df.copy()
    mask_out = pd.Series(False, index=d.index)      # Start with "no rows are outliers"
    for c in cols:
        lo, hi = iqr_bounds(d[c], k)
        col_out = (d[c] < lo) | (d[c] > hi)
        # Update global mask according to the chosen logic
        if how == 'any':
            mask_out = mask_out | col_out        # flag if outlier in ANY selected column
        else:  # 'all'
            mask_out = mask_out & col_out        # flag only if outlier in ALL columns
        #mask_out = (mask_out | col_out) if how=='any' else (mask_out & col_out)
    kept = d.loc[~mask_out].copy()
    removed = d.loc[mask_out].copy()
    return kept, removed

#================================================================
# Model explainability helpers: feature names & importances/coeffs
#================================================================

# Evaluation helpers
def evaluate_classifier(y_true, y_pred, title='Model'):
    print(f'=== {title} ===')
    print('Prediction Accuracy:', round(accuracy_score(y_true, y_pred), 4))
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

def _get_estimator(pipeline):
    """
    Return the final estimator from a sklearn/imblearn Pipeline.
    Prefers a step explicitly named 'model', else falls back to the last step.
    """
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline.named_steps["model"]
    return pipeline.steps[-1][1]

def _get_preprocessor(pipeline):
    """
    Find the ColumnTransformer used for preprocessing.
    Prefers a step named 'prep', otherwise searches for any ColumnTransformer step.
    """
    if hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        return pipeline.named_steps["prep"]
    for name, step in getattr(pipeline, "steps", []):
        if isinstance(step, ColumnTransformer):
            return step
    raise ValueError("No ColumnTransformer found. Expected a 'prep' step or a ColumnTransformer inside the pipeline.")

def _get_feature_names_from_ct(ct: ColumnTransformer, input_feature_names=None):
    """
    Get feature names from a fitted ColumnTransformer.
    Uses get_feature_names_out when available. Falls back to a manual build.
    """
    # Preferred path (sklearn >= 1.0)
    try:
        return ct.get_feature_names_out()
    except Exception:
        pass

    # Fallback: manually build names from transformers_
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name == "remainder" and trans == "passthrough":
            # Pass-through original columns not handled by explicit transformers
            if input_feature_names is not None:
                used = []
                for n2, t2, c2 in ct.transformers_:
                    if c2 not in ("drop", "passthrough"):
                        used.extend(list(c2 if isinstance(c2, (list, tuple)) else [c2]))
                passthru = [c for c in input_feature_names if c not in set(used)]
                names.extend(passthru)
            else:
                names.extend(["remainder"])
            continue

        # Many transformers (e.g., OneHotEncoder) implement get_feature_names_out
        if hasattr(trans, "get_feature_names_out"):
            try:
                fn = trans.get_feature_names_out(cols)
            except Exception:
                fn = trans.get_feature_names_out()
            names.extend(fn)
        elif hasattr(trans, "get_feature_names"):
            fn = trans.get_feature_names()
            names.extend([f"{name}__{f}" for f in fn])
        else:
            # Fallback: just prefix the original column names with the transformer name
            col_list = list(cols) if isinstance(cols, (list, tuple)) else [cols]
            names.extend([f"{name}__{c}" for c in col_list])
    return np.array(names, dtype=object)

def _apply_feature_selector_mask(names, pipeline):
    """
    If you inserted a feature selector step (e.g., 'vt' = VarianceThreshold),
    apply its get_support() mask so names line up with model inputs.
    """
    if hasattr(pipeline, "named_steps") and "vt" in pipeline.named_steps:
        sel = pipeline.named_steps["vt"]
        if hasattr(sel, "get_support"):
            mask = sel.get_support()
            names = np.array(names)[mask]
    return names

def get_feature_names_after_preprocess(pipeline, X_fit_columns=None):
    """
    Return the feature names *after* the preprocessing step inside the pipeline.
    - pipeline: a fitted sklearn/imblearn Pipeline that contains a ColumnTransformer
    - X_fit_columns: original X columns (df.columns). Helps the fallback path.
    """
    ct = _get_preprocessor(pipeline)
    names = _get_feature_names_from_ct(ct, input_feature_names=X_fit_columns)
    names = _apply_feature_selector_mask(names, pipeline)
    return np.array(names, dtype=object)

def feature_importances_dataframe(pipeline, X_fit_columns=None, sort=True):
    """
    Build a DataFrame of feature importances (for tree models) or coefficients (for linear models),
    aligned to the post-preprocessing feature names.

    Returns
    -------
    For trees/ensembles with feature_importances_:
      DataFrame[feature, importance]
    For linear models with coef_ (binary or multiclass):
      - binary: DataFrame[feature, coef, abs_coef]
      - multiclass: DataFrame[feature, coef_<class>..., abs_mean_coef]

    Raises
    ------
    TypeError if the final estimator exposes neither feature_importances_ nor coef_.
    """
    feat_names = get_feature_names_after_preprocess(pipeline, X_fit_columns)
    est = _get_estimator(pipeline)

    # Trees / ensembles
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
        df = pd.DataFrame({"feature": feat_names, "importance": vals})
        return df.sort_values("importance", ascending=False) if sort else df

    # Linear models
    if hasattr(est, "coef_"):
        coef = est.coef_
        if coef.ndim == 1:
            df = pd.DataFrame({"feature": feat_names, "coef": coef})
            df["abs_coef"] = df["coef"].abs()
            return df.sort_values("abs_coef", ascending=False) if sort else df
        else:
            # multiclass: one column per class
            classes = getattr(est, "classes_", [f"class_{i}" for i in range(coef.shape[0])])
            df = pd.DataFrame(coef.T, index=feat_names, columns=[f"coef_{c}" for c in classes]).reset_index()
            df = df.rename(columns={"index": "feature"})
            coef_cols = [c for c in df.columns if c.startswith("coef_")]
            df["abs_mean_coef"] = df[coef_cols].abs().mean(axis=1)
            return df.sort_values("abs_mean_coef", ascending=False) if sort else df

    raise TypeError(
        "Final estimator has no feature_importances_ or coef_. "
        "Examples without importances/coef: KNN, SVC(kernel='rbf'). "
        "Consider sklearn.inspection.permutation_importance if needed."
    )

def plot_top_importances(df_imp: pd.DataFrame, top=20, value_col=None, title="Top features"):
    """
    Convenience plot for the output of feature_importances_dataframe().
    - For trees: value_col='importance'
    - For linear: value_col='abs_coef' (binary) or 'abs_mean_coef' (multiclass)
    """
    if value_col is None:
        # try to pick a sensible default
        for candidate in ["importance", "abs_mean_coef", "abs_coef"]:
            if candidate in df_imp.columns:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError("Could not infer value column. Pass value_col explicitly.")
    dfp = df_imp.sort_values(value_col, ascending=True).tail(top)
    ax = dfp.plot(kind="barh", x="feature", y=value_col, figsize=(7, 6))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    plt.tight_layout()
    plt.show()
    


# ============================================================
# Run a model suite and extract top features/coefficients
# ============================================================

def run_models_with_importances(
    models: dict,
    preprocess,
    X_train, y_train,
    X_test, y_test,
    *,
    top: int = 15,
    plot: bool = True,
    save_csv: bool = False,
    csv_prefix: str = "top_features_",
    pipeline_cls=None,
    sampler=None,
    sampler_name: str = "sampler"
):
    """
    Fit a set of models with a shared preprocessing pipeline, evaluate them,
    and extract top features (tree importances or linear coefficients).

    Parameters
    ----------
    models : dict[str, estimator]
        Mapping of display name -> unfitted sklearn estimator, e.g.
        {'LogisticRegression': LogisticRegression(...), 'DecisionTree': DecisionTreeClassifier(...)}
    preprocess : ColumnTransformer (fitted or unfitted)
        Your preprocessing object (scaler + OHE) to be inserted as ('prep', preprocess).
    X_train, y_train, X_test, y_test : pd.DataFrame/Series
        Train/test data.
    top : int
        How many top features to show/plot.
    plot : bool
        If True, create a horizontal bar chart of the top features.
    save_csv : bool
        If True, save the top-50 features to CSV per model.
    csv_prefix : str
        Prefix for the CSV file names (only used if save_csv=True).
    pipeline_cls : class or None
        Pipeline class to use. Default is sklearn's Pipeline.
        If you pass a sampler (ROS/RUS/SMOTE), set pipeline_cls to imblearn.pipeline.Pipeline.
    sampler : object or None
        Optional resampler inserted between ('prep', ...) and ('model', ...).
        Example: RandomOverSampler(random_state=42)
    sampler_name : str
        Name of the sampler step in the pipeline (default: 'sampler').

    Returns
    -------
    results : dict
        {
          'pipelines': {model_name: fitted_pipeline, ...},
          'importances': {model_name: pd.DataFrame or None, ...}
        }
    """
    # choose pipeline class
    if pipeline_cls is None:
        pipeline_cls = SkPipeline

    results = {"pipelines": {}, "importances": {}}

    for name, est in models.items():
        # Build the pipeline: ('prep', preprocess) -> [optional sampler] -> ('model', est)
        steps = [('prep', preprocess)]
        if sampler is not None:
            steps.append((sampler_name, sampler))
        steps.append(('model', est))
        pipe = pipeline_cls(steps)

        # Fit & evaluate
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # accuracy + confusion + per-class precision/recall/F1
        evaluate_classifier(y_test, y_pred, title=name)

        # Try to get importances/coefficients
        fi_df = None
        try:
            fi_df = feature_importances_dataframe(pipe, X_fit_columns=getattr(X_train, "columns", None))

            # Show top rows in console/notebook
            print(f"\nTop {top} features for {name}:")
            try:
                # Prefer rich display if we're in a notebook
                display(fi_df.head(top))
            except NameError:
                # Fallback if display is not available
                print(fi_df.head(top).to_string(index=False))

            # Pick correct value column for plotting
            value_col = None
            if 'importance' in fi_df.columns:
                value_col = 'importance'          # tree-based
            elif 'abs_mean_coef' in fi_df.columns:
                value_col = 'abs_mean_coef'       # multiclass linear
            elif 'abs_coef' in fi_df.columns:
                value_col = 'abs_coef'            # binary linear

            if plot and value_col is not None:
                plot_top_importances(fi_df, top=top, value_col=value_col, title=f"{name} — top features")

            if save_csv:
                safe = name.replace(" ", "_")
                fi_df.head(top).to_csv(f"{csv_prefix}{safe}.csv", index=False)

        except TypeError:
            # No native importances/coefs (e.g., KNN, RBF SVC)
            print(f"{name}: no native feature_importances_ or coef_ (skipping top-features).\n")

        # Store results
        results["pipelines"][name] = pipe
        results["importances"][name] = fi_df

    return results

