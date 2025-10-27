import os, re, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Student Stress — ML Demo", layout="wide")

# -------- Paths --------
MODELS_DIR   = Path(__file__).parent / "models"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"   # ALL training features saved by the export cell
TESTSET_PATH  = MODELS_DIR / "test_set.csv"

# -------- Loaders (cached) --------
@st.cache_resource
def load_feature_names():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("feature_names.pkl not found. Export from the training notebook first.")
    feats = joblib.load(FEATURES_PATH)
    # enforce list[str]
    feats = [str(f) for f in feats]
    return feats

@st.cache_resource
def load_models():
    models = {}
    if not MODELS_DIR.exists():
        return models
    for p in MODELS_DIR.glob("*.pkl"):
        if p.name == "feature_names.pkl":
            continue
        try:
            models[p.stem] = joblib.load(p)
        except Exception as e:
            st.warning(f"Failed to load {p.name}: {e}")
    return models

@st.cache_resource
def load_testset():
    if TESTSET_PATH.exists():
        try:
            return pd.read_csv(TESTSET_PATH)
        except Exception as e:
            st.warning(f"Could not read test_set.csv: {e}")
    return None

# -------- Helpers --------
def sanitized_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+ -]+", " ", name).strip()

def ensure_int_df(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    # Keep only known columns, correct order, coerce to integers (as requested).
    X = df[feature_names].copy()
    return X.astype("int64", errors="raise")

def predict_single(pipe, feature_names):
    st.subheader("Single Prediction")
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(feature_names):
        with cols[i % 3]:
            # integer-only input
            values[f] = st.number_input(f, value=0, step=1, format="%d")
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([values], columns=feature_names).astype("int64")
        y_pred = pipe.predict(X)[0]
        st.success(f"Predicted stress level: **{int(y_pred)}**")
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)[0]
            st.write("Class probabilities:")
            st.bar_chart(pd.DataFrame(proba, index=[f"class_{i}" for i in range(len(proba))], columns=["prob"]))

def predict_batch(pipe, feature_names):
    st.subheader("Batch Prediction (CSV upload)")
    up = st.file_uploader("Upload CSV with the training feature columns", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())
        try:
            Xb = ensure_int_df(df, feature_names)
        except KeyError as e:
            st.error(f"CSV is missing required columns: {e}")
            return
        except Exception as e:
            st.error(f"Failed to coerce inputs to integers: {e}")
            return
        try:
            preds = pipe.predict(Xb)
            st.write("Predictions (first 20):")
            st.dataframe(pd.DataFrame({"prediction": preds}).head(20))
        except Exception as e:
            st.error(f"Failed to predict: {e}")

def evaluate_on_test(pipe, testset, feature_names, target_col="stress_level"):
    st.subheader("Evaluate on bundled test set")
    try:
        X_eval = ensure_int_df(testset, feature_names)
    except Exception as e:
        st.error(f"Bundled test_set.csv is invalid or missing required columns: {e}")
        return
    if target_col not in testset.columns:
        st.warning(f"Target column '{target_col}' not found in test_set.csv")
        return
    y_true = testset[target_col].values
    try:
        y_pred = pipe.predict(X_eval)
    except Exception as e:
        st.error(f"Failed to predict on test set: {e}")
        return
    acc = accuracy_score(y_true, y_pred)
    st.write(f"**Accuracy:** {acc:.4f}")
    st.text("Classification report:\n" + classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    # st.text("Confusion matrix (rows=actual, cols=pred):\n" + str(cm))
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# -------- Main App --------
def main():
    st.title("🎓 Student Stress — Model Explorer (Integer Inputs)")

    try:
        feature_names = load_feature_names()
    except Exception as e:
        st.error(str(e))
        st.stop()

    models = load_models()
    testset = load_testset()

    if not models:
        st.warning("No models found in models/. Export .pkl files from the training notebook first.")
        st.stop()

    names = sorted(models.keys())
    choice = st.selectbox("Choose a model", names, index=0, format_func=sanitized_name)
    pipe = models[choice]

    col1, col2 = st.columns(2)
    with col1:
        predict_single(pipe, feature_names)
    with col2:
        predict_batch(pipe, feature_names)

    st.divider()
    if testset is not None:
        evaluate_on_test(pipe, testset, feature_names)
    else:
        st.info("No test_set.csv bundled. Export it from the training notebook to enable on-app evaluation.")

if __name__ == "__main__":
    main()
