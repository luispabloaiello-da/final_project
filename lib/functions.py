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

#================================================================
# Library of custom preprocessing functions
#================================================================

# 2) Exclude "binary-like" numerics (e.g., 0/1 flags)
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
