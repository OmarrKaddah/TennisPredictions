"""Shared paths and constants for the ATP prediction pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for d in (RAW_DIR, INTERIM_DIR, OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2000, 2026))
RANKING_DECADES = ("90s", "00s", "10s", "20s", "current")
SACKMANN_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

CLEANED_PARQUET = INTERIM_DIR / "matches_cleaned.parquet"
PREPROCESSED_PARQUET = INTERIM_DIR / "matches_preprocessed.parquet"
FEATURES_PARQUET = OUTPUTS_DIR / "features.parquet"
PLAYER_CLUSTERS_PARQUET = INTERIM_DIR / "player_clusters.parquet"
RANKINGS_PARQUET = INTERIM_DIR / "rankings.parquet"
PLAYERS_PARQUET = INTERIM_DIR / "players.parquet"

KEEP_LEVELS = ("G", "M", "A", "C", "F")
MAIN_DRAW_LEVELS = ("G", "M", "A")

TRAIN_END = 20201231
VAL_END = 20221231
TEST_END = 20251231
TRAIN_START = 20000101
VAL_START = 20210101
TEST_START = 20230101

RANDOM_SEED = 42
ELO_INITIAL = 1500.0
ELO_K = 32.0
GLOBAL_FORM_WINDOW = 20
SURFACE_FORM_WINDOW = 10
SERVE_STATS_WINDOW = 20
FATIGUE_DAYS = 14
FATIGUE_GRAND_SLAM_WEIGHT = 1.5
