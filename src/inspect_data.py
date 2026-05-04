"""Quick inspection helper - prints schema, head, and summary stats for
any intermediate parquet output of the pipeline.

Examples:
    python src/inspect_data.py cleaned
    python src/inspect_data.py preprocessed --rows 20
    python src/inspect_data.py features --cols elo_diff,target,surface
    python src/inspect_data.py clusters
    python src/inspect_data.py predictions --rows 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_PARQUET,
    FEATURES_PARQUET,
    OUTPUTS_DIR,
    PLAYERS_PARQUET,
    PLAYER_CLUSTERS_PARQUET,
    PREPROCESSED_PARQUET,
    RANKINGS_PARQUET,
)


TARGETS = {
    "cleaned": CLEANED_PARQUET,
    "preprocessed": PREPROCESSED_PARQUET,
    "features": FEATURES_PARQUET,
    "clusters": PLAYER_CLUSTERS_PARQUET,
    "predictions": OUTPUTS_DIR / "test_predictions.parquet",
    "rankings": RANKINGS_PARQUET,
    "players": PLAYERS_PARQUET,
}


def load(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files inside {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(TARGETS))
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--cols", default=None, help="comma-separated subset")
    ap.add_argument("--describe", action="store_true", help="numeric summary stats")
    args = ap.parse_args()

    path = TARGETS[args.which]
    if not path.exists():
        print(f"[error] {path} not found - run the producing stage first.")
        return 2

    df = load(path)
    print(f"Source: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    view = df
    if args.cols:
        keep = [c.strip() for c in args.cols.split(",") if c.strip() in df.columns]
        if keep:
            view = df[keep]

    print("\n-- head --")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(view.head(args.rows))

    if args.describe:
        print("\n-- describe --")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(view.describe(include="all").transpose())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
