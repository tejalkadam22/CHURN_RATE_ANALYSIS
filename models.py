"""Model factory, training, K-Fold cross validation and KNN utilities."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             r2_score, mean_absolute_error, mean_squared_error,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, NUMERIC, classification_xy, regression_xy

SEED = 42
MODEL_NAMES = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN"]
SCORING = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
           "F1": "f1", "ROC-AUC": "roc_auc"}


def _preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def build_classifier(name: str, k: int = 11, weights: str = "uniform", metric: str = "minkowski",
                     tree_depth: int = 6, n_trees: int = 150) -> Pipeline:
    if name == "Decision Tree":
        est = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=20,
                                     class_weight="balanced", random_state=SEED)
    elif name == "Random Forest":
        est = RandomForestClassifier(n_estimators=n_trees, max_depth=10, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    elif name == "Logistic Regression":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "KNN":
        est = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    else:
        raise ValueError(name)
    return Pipeline([("prep", _preprocessor(NUMERIC, CATEGORICAL)), ("model", est)])


# ---------------------------------------------------------------- hold-out evaluation
@st.cache_resource(show_spinner="Training models…")
def train_all(df: pd.DataFrame, k: int, weights: str, metric: str, test_size: float):
    X, y = classification_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    out = {}
    for name in MODEL_NAMES:
        pipe = build_classifier(name, k=k, weights=weights, metric=metric).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)
        out[name] = dict(
            pipe=pipe, proba=proba, pred=pred, fpr=fpr, tpr=tpr,
            metrics=dict(
                Accuracy=accuracy_score(y_te, pred), Precision=precision_score(y_te, pred, zero_division=0),
                Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred),
                **{"ROC-AUC": roc_auc_score(y_te, proba)}),
            cm=confusion_matrix(y_te, pred),
        )
    return dict(models=out, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)


# ---------------------------------------------------------------- K-Fold CV
@st.cache_data(show_spinner="Running K-Fold cross-validation…")
def kfold_cv(df: pd.DataFrame, n_splits: int, shuffle: bool, k: int, weights: str, metric: str):
    """Stratified K-Fold CV for every classifier. Returns per-fold scores in long format."""
    X, y = classification_xy(df)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED if shuffle else None)
    rows = []
    for name in MODEL_NAMES:
        pipe = build_classifier(name, k=k, weights=weights, metric=metric)
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        for fold in range(n_splits):
            for label in SCORING:
                rows.append(dict(Model=name, Fold=fold + 1, Metric=label,
                                 Score=res[f"test_{label}"][fold]))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Searching best K for KNN…")
def knn_k_sweep(df: pd.DataFrame, k_max: int, n_splits: int, weights: str, metric: str, scoring: str):
    X, y = classification_xy(df)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for k in range(1, k_max + 1, 2):  # odd k avoids ties
        pipe = build_classifier("KNN", k=k, weights=weights, metric=metric)
        s = cross_val_score(pipe, X, y, cv=cv, scoring=SCORING[scoring], n_jobs=1)
        rows.append(dict(K=k, Mean=s.mean(), Std=s.std()))
    return pd.DataFrame(rows)


def nearest_neighbours(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, row: pd.DataFrame, n: int = 10):
    """Return the n training customers closest to `row` (in the pipeline's preprocessed space)."""
    prep, knn = pipe.named_steps["prep"], pipe.named_steps["model"]
    dist, idx = knn.kneighbors(prep.transform(row), n_neighbors=n)
    nb = X_train.iloc[idx[0]].copy()
    nb.insert(0, "Distance", np.round(dist[0], 3))
    nb.insert(1, "Churned", np.where(y_train.iloc[idx[0]].values == 1, "Yes", "No"))
    return nb


def feature_importance(pipe: Pipeline, top: int = 12) -> pd.DataFrame:
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (pd.DataFrame({"Feature": names, "Importance": vals})
            .sort_values("Importance", ascending=False).head(top).iloc[::-1])


# ---------------------------------------------------------------- regression
@st.cache_resource(show_spinner="Training Linear Regression…")
def train_regression(df: pd.DataFrame):
    X, y = regression_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe = Pipeline([("prep", _preprocessor(["tenure"], CATEGORICAL)), ("model", LinearRegression())]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return dict(pipe=pipe, y_test=y_te, pred=pred,
                r2=r2_score(y_te, pred), mae=mean_absolute_error(y_te, pred),
                rmse=float(np.sqrt(mean_squared_error(y_te, pred))))


@st.cache_data(show_spinner="Cross-validating regression…")
def regression_cv(df: pd.DataFrame, n_splits: int):
    X, y = regression_xy(df)
    pipe = Pipeline([("prep", _preprocessor(["tenure"], CATEGORICAL)), ("model", LinearRegression())])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return pd.DataFrame({"Fold": range(1, n_splits + 1), "R²": r2})
