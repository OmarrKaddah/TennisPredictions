"""Stage 5 - Modeling.

Joins the feature matrix with player cluster labels, splits temporally,
trains Logistic Regression / Random Forest / XGBoost, tunes light
hyperparameter grids on the validation set, and persists the trained
models plus the column schema used at fit time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    FEATURES_PARQUET,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLAYER_CLUSTERS_PARQUET,
    RANDOM_SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


SERVE_DIFF_COLS = [
    "avg_ace_rate_diff",
    "avg_df_rate_diff",
    "avg_1st_serve_pct_diff",
    "avg_1st_serve_win_pct_diff",
    "avg_2nd_serve_win_pct_diff",
    "avg_bp_saved_pct_diff",
]
NUMERIC_FEATURES = [
    "elo_diff",
    "global_form_diff",
    "surface_form_diff",
    "fatigue_diff",
    *SERVE_DIFF_COLS,
    "rank_diff",
    "rank_points_diff",
    "peak_rank_diff",
    "age_diff",
    "height_diff",
    "best_of",
    "round_ordinal",
    "handedness",
]
ONE_HOT_PREFIXES = ("surface_", "tourney_level_", "player1_cluster_", "player2_cluster_")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(str(FEATURES_PARQUET))
    clusters = pd.read_parquet(str(PLAYER_CLUSTERS_PARQUET))[
        ["player_id", "cluster"]
    ]
    p1 = clusters.rename(columns={"player_id": "player1_id", "cluster": "player1_cluster"})
    p2 = clusters.rename(columns={"player_id": "player2_id", "cluster": "player2_cluster"})
    df = df.merge(p1, on="player1_id", how="left")
    df = df.merge(p2, on="player2_id", how="left")

    df["player1_cluster"] = df["player1_cluster"].fillna(-1).astype(int)
    df["player2_cluster"] = df["player2_cluster"].fillna(-1).astype(int)
    df = pd.get_dummies(
        df,
        columns=["player1_cluster", "player2_cluster"],
        prefix=["player1_cluster", "player2_cluster"],
        dtype=int,
    )
    return df


def split_xy(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[pd.DataFrame, pd.Series]:
    return df[feature_cols].copy(), df["target"].astype(int)


def temporal_splits(df: pd.DataFrame):
    train = df[(df["tourney_date"] >= TRAIN_START) & (df["tourney_date"] <= TRAIN_END)]
    val = df[(df["tourney_date"] >= VAL_START) & (df["tourney_date"] <= VAL_END)]
    test = df[(df["tourney_date"] >= TEST_START) & (df["tourney_date"] <= TEST_END)]
    return train, val, test


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
    }


def fit_logistic(Xtr, ytr, Xval, yval, feature_cols):
    Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
    Xval = Xval.fillna(Xtr.median(numeric_only=True))
    best = None
    for C in (0.01, 0.1, 1.0, 3.0, 10.0):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xval)[:, 1]
        m = metrics(yval, p)
        print(f"  LR  C={C:>6}  val={m}")
        if best is None or m["log_loss"] < best[1]["log_loss"]:
            best = (pipe, m, {"C": C})
    return best


def fit_random_forest(Xtr, ytr, Xval, yval):
    Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
    Xval = Xval.fillna(Xtr.median(numeric_only=True))
    best = None
    grid = [
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 3},
        {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2},
    ]
    for params in grid:
        clf = RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **params,
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xval)[:, 1]
        m = metrics(yval, p)
        print(f"  RF  {params}  val={m}")
        if best is None or m["log_loss"] < best[1]["log_loss"]:
            best = (clf, m, params)
    return best


def fit_xgboost(Xtr, ytr, Xval, yval):
    best = None
    grid = [
        {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.9},
        {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8},
        {"n_estimators": 2000, "learning_rate": 0.02, "max_depth": 5, "subsample": 0.85},
    ]
    for params in grid:
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
            **params,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        p = clf.predict_proba(Xval)[:, 1]
        m = metrics(yval, p)
        print(f"  XGB {params}  val={m}  best_iter={clf.best_iteration}")
        if best is None or m["log_loss"] < best[1]["log_loss"]:
            best = (clf, m, params)
    return best


def main() -> int:
    if not FEATURES_PARQUET.exists():
        print(f"[error] missing {FEATURES_PARQUET}. Run: python src/03_feature_engineering.py")
        return 2
    if not PLAYER_CLUSTERS_PARQUET.exists():
        print(f"[error] missing {PLAYER_CLUSTERS_PARQUET}. Run: python src/04_clustering.py")
        return 2

    df = load_features()
    print(f"Feature matrix: {df.shape}")
    print(f"Class balance overall: {df['target'].mean():.3f}")

    one_hot_cols = sorted(
        [c for c in df.columns if c.startswith(ONE_HOT_PREFIXES)
         and c not in NUMERIC_FEATURES]
    )
    feature_cols = NUMERIC_FEATURES + one_hot_cols
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))  # guard against any remaining duplicates
    print(f"Total features: {len(feature_cols)}")

    train, val, test = temporal_splits(df)
    print(f"Train {len(train):,}  Val {len(val):,}  Test {len(test):,}")

    Xtr, ytr = split_xy(train, feature_cols)
    Xval, yval = split_xy(val, feature_cols)
    Xte, yte = split_xy(test, feature_cols)

    Xtr = Xtr.astype(float)
    Xval = Xval.astype(float)
    Xte = Xte.astype(float)

    print("\n== Logistic Regression ==")
    lr_model, lr_val, lr_params = fit_logistic(Xtr, ytr, Xval, yval, feature_cols)
    print("\n== Random Forest ==")
    rf_model, rf_val, rf_params = fit_random_forest(Xtr, ytr, Xval, yval)
    print("\n== XGBoost ==")
    xgb_model, xgb_val, xgb_params = fit_xgboost(Xtr, ytr, Xval, yval)

    median = Xtr.median(numeric_only=True)
    Xte_lr = Xte.fillna(median)
    Xte_rf = Xte.fillna(median)

    lr_p = lr_model.predict_proba(Xte_lr)[:, 1]
    rf_p = rf_model.predict_proba(Xte_rf)[:, 1]
    xgb_p = xgb_model.predict_proba(Xte)[:, 1]

    test_metrics = {
        "logistic_regression": metrics(yte, lr_p),
        "random_forest": metrics(yte, rf_p),
        "xgboost": metrics(yte, xgb_p),
    }
    val_metrics = {
        "logistic_regression": lr_val,
        "random_forest": rf_val,
        "xgboost": xgb_val,
    }
    chosen = {
        "logistic_regression": lr_params,
        "random_forest": rf_params,
        "xgboost": xgb_params,
    }
    print("\n== Validation metrics ==")
    print(json.dumps(val_metrics, indent=2))
    print("\n== Test metrics ==")
    print(json.dumps(test_metrics, indent=2))

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")
    xgb_model.save_model(str(MODELS_DIR / "xgboost.json"))
    joblib.dump(median, MODELS_DIR / "train_median.joblib")

    schema = {
        "feature_cols": feature_cols,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "chosen_params": chosen,
        "splits": {
            "train": [TRAIN_START, TRAIN_END, len(train)],
            "val": [VAL_START, VAL_END, len(val)],
            "test": [TEST_START, TEST_END, len(test)],
        },
    }
    schema_path = OUTPUTS_DIR / "model_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Wrote {schema_path}")

    test_predictions = test[
        ["tourney_date", "surface", "tourney_level", "round", "target", "elo_diff"]
    ].copy()
    test_predictions["lr_prob"] = lr_p
    test_predictions["rf_prob"] = rf_p
    test_predictions["xgb_prob"] = xgb_p
    pred_path = OUTPUTS_DIR / "test_predictions.parquet"
    test_predictions.to_parquet(pred_path, index=False)
    print(f"Wrote {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
