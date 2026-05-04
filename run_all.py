"""End-to-end pipeline runner.

Invokes each stage in order. Skips stages whose output already exists
unless --force is passed. Stage scripts are referenced by file path
because their numeric prefixes are not valid Python identifiers.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))
from config import (
    CLEANED_PARQUET,
    FEATURES_PARQUET,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLAYER_CLUSTERS_PARQUET,
    PREPROCESSED_PARQUET,
    RAW_DIR,
)


STAGES = [
    ("download", SRC / "00_download_data.py", lambda: any(RAW_DIR.glob("atp_matches_*.csv"))),
    ("ingestion", SRC / "01_ingestion.py", lambda: CLEANED_PARQUET.exists()),
    ("preprocessing", SRC / "02_preprocessing.py", lambda: PREPROCESSED_PARQUET.exists()),
    ("features", SRC / "03_feature_engineering.py", lambda: FEATURES_PARQUET.exists()),
    ("clustering", SRC / "04_clustering.py", lambda: PLAYER_CLUSTERS_PARQUET.exists()),
    ("modeling", SRC / "05_modeling.py", lambda: (MODELS_DIR / "xgboost.json").exists()),
    ("evaluation", SRC / "06_evaluation.py", lambda: (OUTPUTS_DIR / "evaluation_summary.json").exists()),
]


def run_stage(name: str, script: Path) -> int:
    print(f"\n=========== {name} ({script.name}) ===========")
    try:
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun all stages even if outputs exist")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=[s[0] for s in STAGES],
        help="run only the named stages",
    )
    args = ap.parse_args()

    for name, script, done_check in STAGES:
        if args.only and name not in args.only:
            continue
        if not args.force and done_check():
            print(f"[skip] {name} already produced output")
            continue
        rc = run_stage(name, script)
        if rc != 0:
            print(f"[fail] {name} returned {rc}")
            return rc
    print("\nAll stages complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
