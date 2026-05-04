# ATP Tennis Match Prediction

Predicts the winner of ATP
matches using surface-specific Elo, rolling form, fatigue, rolling
serve stats, K-Means style clusters, and XGBoost. See
[atp_tennis_prediction_plan.md](atp_tennis_prediction_plan.md) for the
full design.

## Data scope (~113 MB processed, satisfies the 100 MB course requirement)

| Dataset | Files | Size | Use |
|---|---|---|---|
| `atp_matches_YYYY.csv` (main draw, 2000-2024) | 25 | ~14.6 MB | prediction targets |
| `atp_matches_qual_chall_YYYY.csv` (challenger + qualifying, 2000-2024) | 25 | ~32.2 MB | feed into player Elo / form / serve history (not predicted on) |
| `atp_rankings_{90s,00s,10s,20s,current}.csv` | 5 | ~64 MB | weekly rankings, ingested via Spark |
| `atp_players.csv` | 1 | ~2.4 MB | player metadata |

The challenger/qualifying file shares the exact 49-column schema with
the main file, so we union them and tag every row with
`match_source` ('main' or 'qual_chall'). Stage 3 walks every match
chronologically to update each player's Elo / form / serve-stats
state, but only emits **prediction rows where `match_source = 'main'`
AND `tourney_level in (G, M, A)`**. This way Elo and rolling stats
are informed by far more matches per player while predictions still
target only the ATP main tour.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Java 8/11/17 is required for PySpark. Set `JAVA_HOME` if PySpark fails
to launch.

## Run

```powershell
python run_all.py
```

This executes every stage in order, skipping any whose output already
exists. Use `--force` to rerun everything or `--only <stage>` to run a
subset.

Stages:

| # | Script | Output |
|---|--------|--------|
| 0 | [src/00_download_data.py](src/00_download_data.py) | `data/raw/atp_matches_YYYY.csv` |
| 1 | [src/01_ingestion.py](src/01_ingestion.py) | `data/interim/matches_cleaned.parquet` |
| 2 | [src/02_preprocessing.py](src/02_preprocessing.py) | `data/interim/matches_preprocessed.parquet` |
| 3 | [src/03_feature_engineering.py](src/03_feature_engineering.py) | `outputs/features.parquet` |
| 4 | [src/04_clustering.py](src/04_clustering.py) | `data/interim/player_clusters.parquet`, `outputs/plots/kmeans_elbow.png` |
| 5 | [src/05_modeling.py](src/05_modeling.py) | `outputs/models/*.{joblib,json}`, `outputs/test_predictions.parquet`, `outputs/model_schema.json` |
| 6 | [src/06_evaluation.py](src/06_evaluation.py) | `outputs/metrics_by_*.csv`, `outputs/evaluation_summary.json`, `outputs/plots/*.png` |

## Running and debugging stages individually

Every stage is a standalone script. Each one checks for its expected
input parquet and prints a clear `[error] ... Run: python src/XX_*.py`
message if a prerequisite is missing - so you can run them in any
order and catch what's missing:

```powershell
python src/00_download_data.py        # 26 CSVs into data/raw/
python src/01_ingestion.py            # union + filter -> matches_cleaned.parquet
python src/02_preprocessing.py        # serve %s + null fills -> matches_preprocessed.parquet
python src/03_feature_engineering.py  # Elo, form, fatigue, serve roll, row flip -> features.parquet
python src/04_clustering.py           # K-Means -> player_clusters.parquet + elbow plot
python src/05_modeling.py             # LR / RF / XGB -> models/ + test_predictions.parquet
python src/06_evaluation.py           # tables + plots -> outputs/ + plots/
```

To peek at any intermediate output without writing code:

```powershell
python src/inspect_data.py cleaned
python src/inspect_data.py preprocessed --rows 5
python src/inspect_data.py features --cols target,elo_diff,surface --describe
python src/inspect_data.py clusters
python src/inspect_data.py predictions --rows 20
python src/inspect_data.py rankings --rows 5
python src/inspect_data.py players --rows 5
```

## Splits

Temporal, never shuffled:

- Train: 2000-01-01 to 2020-12-31
- Validation: 2021-01-01 to 2022-12-31
- Test: 2023-01-01 to 2025-12-31

## Notes

Sequential computations (Elo, rolling windows, fatigue) run in pandas
on the driver after Spark loads the cleaned parquet. The dataset is
~70k rows, which fits comfortably in memory; pure Spark window
functions cannot express the per-player Elo update because each match
depends on the previous one.

## Citation

Jeff Sackmann / Tennis Abstract, `tennis_atp` dataset
(<https://github.com/JeffSackmann/tennis_atp>). Licensed CC BY-NC-SA
4.0. Tennis Elo methodology: Jeff Sackmann, "An Introduction to
Tennis Elo," Tennis Abstract, December 2019.
