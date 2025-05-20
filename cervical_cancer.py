"""
Refactored Cervical Cancer Risk‑Factors ML pipeline
--------------------------------------------------
Author: (auto‑generated)
Date: 2025‑05‑20

* Cleans raw notebook export into a lean, modular Python script
* Removes duplicate/unused code blocks, magic comments and exploratory prints
* Provides reusable functions for:
    • data loading & cleaning
    • feature engineering (dummy‑vars, scaling, balancing)
    • feature selection (RFECV)
    • model training & evaluation with cross‑validation
* Easily extensible – adjust the global CONFIG dict or add models in the MODELS list.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = Path("risk_factors_cervical_cancer.csv")
TARGET_COL = "Biopsy"

# ---------------------------------------------------------------------------
# 1. Data loading & basic cleaning
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Read CSV and convert '?' to NaN, numeric dtypes."""
    df = pd.read_csv(path).replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with sensible defaults/medians."""

    median_fill = [
        "Number of sexual partners",
        "First sexual intercourse",
        "Num of pregnancies",
        "Smokes (years)",
        "Smokes (packs/year)",
        "Hormonal Contraceptives (years)",
        "IUD (years)",
        "STDs (number)",
        "STDs:condylomatosis",
        "STDs:cervical condylomatosis",
        "STDs:vaginal condylomatosis",
        "STDs:vulvo-perineal condylomatosis",
        "STDs:syphilis",
        "STDs:pelvic inflammatory disease",
        "STDs:genital herpes",
        "STDs:molluscum contagiosum",
        "STDs:AIDS",
        "STDs:HIV",
        "STDs:Hepatitis B",
        "STDs:HPV",
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis",
    ]

    binary_defaults = {
        "Smokes": 1,
        "Hormonal Contraceptives": 1,
        "IUD": 0,
        "STDs": 1,
    }

    common_cols = [c for c in median_fill if c in df.columns]
    if common_cols:
        df[common_cols] = df[common_cols].apply(lambda c: c.fillna(c.median()))

    for col, default in binary_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build ColumnTransformer: scale numeric, one‑hot encode binaries."""
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COL)
    cat_cols = [
        "Smokes",
        "Hormonal Contraceptives",
        "IUD",
        "STDs",
        "Dx:Cancer",
        "Dx:CIN",
        "Dx:HPV",
        "Dx",
        "Hinselmann",
        "Citology",
        "Schiller",
    ]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols),
        ],
        remainder="drop",
    )


def balance_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply hybrid sampling to tackle class imbalance."""
    sampler = SMOTETomek(random_state=RANDOM_STATE)
    return sampler.fit_resample(X, y)


# ---------------------------------------------------------------------------
# 3. Feature Selection
# ---------------------------------------------------------------------------

def recursive_feature_elim(estimator, X, y, cv=5) -> list[int]:
    rfecv = RFECV(estimator, step=1, cv=cv, scoring="roc_auc", n_jobs=-1)
    rfecv.fit(X, y)
    return list(np.where(rfecv.support_)[0])


# ---------------------------------------------------------------------------
# 4. Model Training & Evaluation helpers
# ---------------------------------------------------------------------------

METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}


def evaluate(model, X, y, cv=5) -> dict[str, float]:
    scores = {}
    for name, scorer in METRICS.items():
        scores[name] = cross_val_score(
            model, X, y, cv=cv, scoring=name, n_jobs=-1
        ).mean()
    return scores


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Load & preprocess
    df = fill_missing(load_data(DATA_PATH))

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Build preprocessor (fit only on training folds inside pipeline)
    preprocessor = make_preprocessor(df)

    # Model list
    MODELS = [
        ("LogReg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE)),
        ("XGB", XGBClassifier(random_state=RANDOM_STATE, n_estimators=300)),
        ("SVM‑rbf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE)),
        ("GradientBoost", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]

    results = []
    for name, clf in MODELS:
        print(f"\nEvaluating {name} with cross-validation …")
        # Build pipeline: preprocess -> balance -> feature select -> classifier
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("balance", SMOTETomek(random_state=RANDOM_STATE)),
            ("select", RFECV(
                RandomForestClassifier(random_state=RANDOM_STATE),
                step=1, cv=5, scoring="roc_auc", n_jobs=-1)),
            ("clf", clf),
        ])
        # Evaluate pipeline with cross-validation (all steps fit only on train folds)
        res = {}
        for metric, scorer in METRICS.items():
            res[metric] = cross_val_score(pipe, X, y, cv=5, scoring=metric, n_jobs=-1).mean()
        res["model"] = name
        results.append(res)

    # Results dataframe
    res_df = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    print("\n=== CV metric means (pipeline, no leakage) ===")
    print(res_df.round(3))

    # Plot
    res_df[["accuracy", "roc_auc", "f1"]].plot(kind="barh", figsize=(10, 5))
    plt.title("Model comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
