"""Stage 6 - Evaluation & Insights.

Reads the test predictions and trained XGBoost model, then writes:
- surface-stratified metric tables
- accuracy by tourney_level
- feature importance plot
- calibration curve
- confusion matrix
- top notable upsets in the test set
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    FEATURES_PARQUET,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLAYER_CLUSTERS_PARQUET,
    PLOTS_DIR,
    TEST_END,
    TEST_START,
)


def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(OUTPUTS_DIR / "test_predictions.parquet")


def load_schema() -> dict:
    return json.loads((OUTPUTS_DIR / "model_schema.json").read_text())


def metric_row(name: str, y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "slice": name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p)) if len(set(y)) > 1 else float("nan"),
        "auc_roc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
    }


def stratified_table(df: pd.DataFrame, by: str, prob_col: str) -> pd.DataFrame:
    rows = []
    for val, sub in df.groupby(by):
        rows.append(metric_row(f"{by}={val}", sub["target"], sub[prob_col]))
    rows.append(metric_row("overall", df["target"], df[prob_col]))
    return pd.DataFrame(rows)


def plot_feature_importance(model: XGBClassifier, feature_cols: list[str]) -> Path:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    name_map = {f"f{i}": c for i, c in enumerate(feature_cols)}
    imp = pd.Series(
        {name_map.get(k, k): v for k, v in score.items()}
    ).sort_values(ascending=True)
    imp = imp.tail(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#3b82f6")
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost - Top 20 features by gain")
    fig.tight_layout()
    out = PLOTS_DIR / "feature_importance_xgb.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_calibration(df: pd.DataFrame, prob_col: str) -> Path:
    frac_pos, mean_pred = calibration_curve(df["target"], df[prob_col], n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="XGBoost")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of player1 wins")
    ax.set_title("Calibration curve - test set")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_confusion(df: pd.DataFrame, prob_col: str) -> Path:
    yhat = (df[prob_col] >= 0.5).astype(int)
    cm = confusion_matrix(df["target"], yhat)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred player2 win", "pred player1 win"])
    ax.set_yticklabels(["actual player2 win", "actual player1 win"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    ax.set_title("Confusion matrix - test set")
    fig.colorbar(im)
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_accuracy_by_level(df: pd.DataFrame, prob_col: str) -> Path:
    rows = []
    for lvl, sub in df.groupby("tourney_level"):
        yhat = (sub[prob_col] >= 0.5).astype(int)
        rows.append({"tourney_level": lvl, "accuracy": accuracy_score(sub["target"], yhat), "n": len(sub)})
    summary = pd.DataFrame(rows).sort_values("tourney_level")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["tourney_level"], summary["accuracy"], color="#10b981")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by tourney_level (test)")
    for x, acc, n in zip(summary["tourney_level"], summary["accuracy"], summary["n"]):
        ax.text(x, acc + 0.01, f"{acc:.2f}\n(n={n})", ha="center")
    fig.tight_layout()
    out = PLOTS_DIR / "accuracy_by_level.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_roc_curves(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for col, label, color in [
        ("lr_prob",  "Logistic Regression", "#3b82f6"),
        ("rf_prob",  "Random Forest",       "#10b981"),
        ("xgb_prob", "XGBoost",             "#f59e0b"),
    ]:
        fpr, tpr, _ = roc_curve(df["target"], df[col])
        auc = roc_auc_score(df["target"], df[col])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / "roc_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_temporal_accuracy(df: pd.DataFrame, prob_col: str) -> Path:
    df = df.copy()
    df["year"] = (df["tourney_date"] // 10000).astype(int)
    rows = []
    for yr, sub in df.groupby("year"):
        if len(sub) < 10:
            continue
        yhat = (sub[prob_col] >= 0.5).astype(int)
        rows.append({
            "year": yr,
            "accuracy": accuracy_score(sub["target"], yhat),
            "auc_roc": roc_auc_score(sub["target"], sub[prob_col]),
            "n": len(sub),
        })
    tbl = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.bar(tbl["year"].astype(str), tbl["accuracy"], color="#3b82f6", alpha=0.7, label="Accuracy")
    ax2.plot(tbl["year"].astype(str), tbl["auc_roc"], color="#ef4444", marker="o", label="AUC-ROC")
    ax1.set_ylim(0.5, 0.85)
    ax2.set_ylim(0.5, 0.90)
    ax1.set_ylabel("Accuracy", color="#3b82f6")
    ax2.set_ylabel("AUC-ROC", color="#ef4444")
    ax1.set_xlabel("Year")
    ax1.set_title("XGBoost performance by year — test set")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / "temporal_accuracy.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_elo_diff_win_rate(df: pd.DataFrame, prob_col: str) -> Path:
    df = df.copy()
    df["elo_bin"] = pd.cut(df["elo_diff"], bins=10)
    grouped = df.groupby("elo_bin", observed=True).agg(
        actual_win_rate=("target", "mean"),
        mean_predicted=  (prob_col, "mean"),
        n=               ("target", "count"),
    ).reset_index()
    grouped["bin_mid"] = grouped["elo_bin"].apply(lambda x: x.mid)
    grouped = grouped[grouped["n"] >= 20]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(grouped["bin_mid"], grouped["actual_win_rate"],
               s=grouped["n"] / grouped["n"].max() * 200,
               label="Actual win rate", color="#3b82f6", zorder=3)
    ax.plot(grouped["bin_mid"], grouped["mean_predicted"],
            color="#f59e0b", linewidth=2, label="Mean predicted prob", zorder=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Elo diff (player1 − player2)")
    ax.set_ylabel("Win rate / predicted probability")
    ax.set_title("Elo diff vs actual win rate — test set\n(bubble size ∝ match count)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "elo_diff_win_rate.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_upset_distribution(df: pd.DataFrame, prob_col: str) -> Path:
    upset_mask = ((df["target"] == 1) & (df[prob_col] < 0.5)) | (
        (df["target"] == 0) & (df[prob_col] > 0.5)
    )
    upset_probs = np.where(
        df["target"] == 1, 1 - df[prob_col], df[prob_col]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(upset_probs[upset_mask], bins=30, color="#ef4444", alpha=0.85)
    ax.set_xlabel("Predicted probability for the actual loser")
    ax.set_ylabel("Match count")
    ax.set_title("Predicted probability distribution for upsets (test set)")
    fig.tight_layout()
    out = PLOTS_DIR / "upset_probability_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    pred_path = OUTPUTS_DIR / "test_predictions.parquet"
    schema_path = OUTPUTS_DIR / "model_schema.json"
    if not pred_path.exists() or not schema_path.exists():
        print(f"[error] missing {pred_path} or {schema_path}. Run: python src/05_modeling.py")
        return 2

    schema = load_schema()
    feature_cols = schema["feature_cols"]
    pred = load_predictions()
    print(f"Test predictions: {len(pred):,} rows")

    surface_table = stratified_table(pred, "surface", "xgb_prob")
    level_table = stratified_table(pred, "tourney_level", "xgb_prob")
    surface_path = OUTPUTS_DIR / "metrics_by_surface.csv"
    level_path = OUTPUTS_DIR / "metrics_by_level.csv"
    surface_table.to_csv(surface_path, index=False)
    level_table.to_csv(level_path, index=False)
    print("\nMetrics by surface:")
    print(surface_table.to_string(index=False))
    print("\nMetrics by tourney_level:")
    print(level_table.to_string(index=False))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost.json"))

    paths = []
    paths.append(plot_feature_importance(xgb_model, feature_cols))
    paths.append(plot_calibration(pred, "xgb_prob"))
    paths.append(plot_confusion(pred, "xgb_prob"))
    paths.append(plot_accuracy_by_level(pred, "xgb_prob"))
    paths.append(plot_upset_distribution(pred, "xgb_prob"))
    paths.append(plot_roc_curves(pred))
    paths.append(plot_temporal_accuracy(pred, "xgb_prob"))
    if "elo_diff" in pred.columns:
        paths.append(plot_elo_diff_win_rate(pred, "xgb_prob"))
    for p in paths:
        print(f"Wrote {p}")

    summary = {
        "metrics_by_surface": surface_table.to_dict(orient="records"),
        "metrics_by_level": level_table.to_dict(orient="records"),
        "test_metrics_all_models": schema["test_metrics"],
        "validation_metrics_all_models": schema["val_metrics"],
        "chosen_params": schema["chosen_params"],
        "splits": schema["splits"],
        "unsuccessful_trials": [
            "Neural Network (MLP): excluded - underperforms tree models on this tabular feature set and is slower to train.",
            "Removed any feature whose addition increased validation log_loss; track here in future iterations.",
        ],
    }
    summary_path = OUTPUTS_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
