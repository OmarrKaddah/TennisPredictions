"""Stage 4 - Player Style Clustering (K-Means).

Computes career-average serve stats per player, runs K-Means for K=2..8,
plots inertia (elbow), picks K automatically via the kneedle-style
heuristic (largest distance from the line connecting the endpoints),
and writes a player_id -> cluster mapping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    PLAYER_CLUSTERS_PARQUET,
    PLOTS_DIR,
    PREPROCESSED_PARQUET,
    RANDOM_SEED,
)
from spark_session import get_spark


SERVE_COLUMNS = (
    "ace_rate",
    "df_rate",
    "1st_serve_pct",
    "1st_serve_win_pct",
    "2nd_serve_win_pct",
    "bp_saved_pct",
)


def load_career_averages() -> pd.DataFrame:
    spark = get_spark("atp_clustering_load")
    spark.sparkContext.setLogLevel("WARN")
    pdf = spark.read.parquet(str(PREPROCESSED_PARQUET)).toPandas()
    spark.stop()

    parts = []
    for side, id_col, name_col in [
        ("w", "winner_id", "winner_name"),
        ("l", "loser_id", "loser_name"),
    ]:
        cols = {f"{side}_{c}": c for c in SERVE_COLUMNS}
        sub = pdf[[id_col, name_col, *cols.keys()]].rename(
            columns={id_col: "player_id", name_col: "player_name", **cols}
        )
        parts.append(sub)
    long = pd.concat(parts, ignore_index=True)
    grouped = (
        long.groupby("player_id", as_index=False)
        .agg(
            {
                **{c: "mean" for c in SERVE_COLUMNS},
                "player_name": "last",
            }
        )
    )
    grouped = grouped.dropna(subset=list(SERVE_COLUMNS), how="any")
    return grouped


def pick_k_elbow(ks: list[int], inertias: list[float]) -> int:
    """Pick the K with the largest perpendicular distance from the line
    connecting (k_min, inertia_min) and (k_max, inertia_max)."""
    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    best_k = ks[0]
    best_dist = -1.0
    for k, inertia in zip(ks, inertias):
        p = np.array([k, inertia])
        cross = abs(np.cross(line_vec, p - p1))
        dist = cross / line_len if line_len else 0.0
        if dist > best_dist:
            best_dist = dist
            best_k = k
    return int(best_k)


def main() -> int:
    if not PREPROCESSED_PARQUET.exists():
        print(f"[error] missing {PREPROCESSED_PARQUET}. Run: python src/02_preprocessing.py")
        return 2

    df = load_career_averages()
    print(f"Players with full serve profile: {len(df):,}")

    X = df[list(SERVE_COLUMNS)].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ks = list(range(2, 9))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        km.fit(Xs)
        inertias.append(float(km.inertia_))
        print(f"K={k}  inertia={km.inertia_:.1f}")

    k_opt = pick_k_elbow(ks, inertias)
    print(f"Selected K={k_opt} via elbow heuristic")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, inertias, marker="o")
    ax.axvline(k_opt, color="red", linestyle="--", label=f"selected K={k_opt}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow - Player Serve Profiles")
    ax.legend()
    fig.tight_layout()
    elbow_path = PLOTS_DIR / "kmeans_elbow.png"
    fig.savefig(elbow_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {elbow_path}")

    final = KMeans(n_clusters=k_opt, n_init=10, random_state=RANDOM_SEED)
    labels = final.fit_predict(Xs)

    centers = pd.DataFrame(
        scaler.inverse_transform(final.cluster_centers_),
        columns=list(SERVE_COLUMNS),
    )
    centers.insert(0, "cluster", range(k_opt))
    print("Cluster centroids (un-scaled):")
    print(centers.round(3).to_string(index=False))

    df_out = df[["player_id", "player_name"]].copy()
    df_out["cluster"] = labels.astype(int)
    df_out.to_parquet(str(PLAYER_CLUSTERS_PARQUET), index=False)
    print(f"Wrote {PLAYER_CLUSTERS_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
