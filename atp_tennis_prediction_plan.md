# ATP Tennis Match Prediction — Implementation Plan
## CMPS451 Big Data & Analytics Project

---

## Stack
- **PySpark** — data ingestion, preprocessing, feature computation
- **Python / pandas** — model training (after exporting features from Spark)
- **scikit-learn** — Logistic Regression, Random Forest
- **XGBoost** — final model
- **matplotlib / seaborn** — visualization

---

## Dataset
**Source:** Jeff Sackmann's `tennis_atp` GitHub repository
`https://github.com/JeffSackmann/tennis_atp`

**Files to download:** `atp_matches_YYYY.csv` for every year 2000–2025

**Columns used:**
| Column | Description |
|---|---|
| `tourney_date` | Tournament start date (YYYYMMDD integer) |
| `tourney_level` | G = Grand Slam, M = Masters, A = ATP 500/250 |
| `surface` | Hard / Clay / Grass / Carpet |
| `round` | R128, R64, R32, R16, QF, SF, F |
| `best_of` | 3 or 5 |
| `winner_id / loser_id` | Player IDs |
| `winner_name / loser_name` | Player names |
| `winner_rank / loser_rank` | ATP ranking at tournament start |
| `winner_rank_points / loser_rank_points` | Ranking points |
| `winner_age / loser_age` | Age at tournament start |
| `winner_ht / loser_ht` | Height in cm |
| `winner_hand / loser_hand` | R = Right, L = Left |
| `w_ace, l_ace` | Aces served |
| `w_df, l_df` | Double faults |
| `w_svpt, l_svpt` | Total service points |
| `w_1stIn, l_1stIn` | First serves in |
| `w_1stWon, l_1stWon` | Points won on first serve |
| `w_2ndWon, l_2ndWon` | Points won on second serve |
| `w_SvGms, l_SvGms` | Service games played |
| `w_bpSaved, l_bpSaved` | Break points saved |
| `w_bpFaced, l_bpFaced` | Break points faced |
| `minutes` | Match duration (has nulls) |

---

## Stage 1 — Data Ingestion (PySpark)

1. Initialize Spark session in pseudo-distributed mode
2. Load all 26 CSV files (2000–2025) using `spark.read.csv()` with `header=True`, `inferSchema=True`
3. Union all DataFrames into one using `reduce(DataFrame.union, dfs)`
4. Filter to keep only `tourney_level` IN ('G', 'M', 'A') — drop Davis Cup (D) and other levels
5. Filter out retirements and walkovers:
   - Drop rows where `score` contains 'RET', 'W/O', 'DEF', or 'ABN'
6. Cast `tourney_date` to integer, sort ascending by `tourney_date` then `match_num`
7. Cache the cleaned DataFrame

**Output:** Single sorted Spark DataFrame, ~60,000–70,000 rows

---

## Stage 2 — Preprocessing

### 2a. Derive serve percentage columns (from raw integer totals)
Compute these for both winner (w_) and loser (l_):

```
1st_serve_pct     = w_1stIn / w_svpt
1st_serve_win_pct = w_1stWon / w_1stIn
2nd_serve_win_pct = w_2ndWon / (w_svpt - w_1stIn)
bp_saved_pct      = w_bpSaved / w_bpFaced
ace_rate          = w_ace / w_svpt
df_rate           = w_df / w_svpt
```

Guard against division by zero — use `when(denominator > 0, numerator/denominator).otherwise(None)`

### 2b. Row flip — create balanced training data
**Critical:** The dataset always puts the winner in winner columns. If you train on this naively, the model learns nothing — it just learns that "player 1" always wins.

For each row, randomly assign winner/loser as player1/player2:
- With 50% probability: player1 = winner, player2 = loser, target = 1
- With 50% probability: player1 = loser, player2 = winner, target = 0

Use a fixed random seed for reproducibility.

**Important:** Do all historical feature computation (Elo, form, serve stats) BEFORE the flip, on the original winner/loser structure. Join features back after flipping.

### 2c. Handle nulls
- `winner_rank / loser_rank`: fill nulls with a high rank (e.g. 999) — unranked players are unknown quantities
- `winner_ht / loser_ht`: fill with surface/era median
- Serve stat columns: leave as null for now — XGBoost handles natively; for Logistic Regression impute with player career mean

### 2d. Temporal train/val/test split
Split on `tourney_date` (integer YYYYMMDD):
- **Train:** 20000101 – 20201231
- **Validation:** 20210101 – 20221231
- **Test:** 20230101 – 20251231

Never shuffle across these boundaries.

---

## Stage 3 — Feature Engineering

All features are computed from historical data only — never using information from the current match. All are expressed as `player1_value - player2_value` (difference) so the sign carries meaning.

### 3a. Surface-specific Elo rating

Process all matches chronologically. Before each match, record both players' current Elo on that surface. Then update after.

**Initialization:** Every player starts at 1500 on each surface.

**Expected score formula:**
```
E1 = 1 / (1 + 10^((R2 - R1) / 400))
```

**Update formula:**
```
R1_new = R1 + K * (S1 - E1)
R2_new = R2 + K * (S2 - E2)
```
Where:
- K = 32
- S1 = 1 if player 1 won, 0 if lost
- S2 = 1 - S1

Maintain four separate dictionaries: `elo_hard`, `elo_clay`, `elo_grass`, `elo_carpet`

**Feature:** `elo_diff = player1_elo_surface - player2_elo_surface`

Source: Jeff Sackmann, Tennis Abstract — "An Introduction to Tennis Elo" (2019)
`https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/`

### 3b. Global rolling win rate (last 20 matches)
For each player at each match date, compute their win rate over their previous 20 matches regardless of surface.

**Feature:** `global_form_diff = player1_global_form - player2_global_form`

### 3c. Surface-specific rolling win rate (last 10 matches on same surface)
Same as above but filtered to the same surface as the current match. Window = 10 to account for sparsity (grass players may only play 8 grass matches per year).

**Feature:** `surface_form_diff = player1_surface_form - player2_surface_form`

### 3d. Fatigue index (matches in last 14 days)
Count matches each player played in the 14 days before this match's `tourney_date`.
Weight Grand Slam matches as 1.5 (best-of-5 is physically heavier).

```
fatigue = sum(1.5 if match_level == 'G' else 1.0 for each match in last 14 days)
```

**Feature:** `fatigue_diff = player1_fatigue - player2_fatigue`

### 3e. Rolling historical serve stats (last 20 matches)
For each player, compute rolling 20-match averages of:
- `avg_ace_rate`
- `avg_df_rate`
- `avg_1st_serve_pct`
- `avg_1st_serve_win_pct`
- `avg_2nd_serve_win_pct`
- `avg_bp_saved_pct`

**Important:** Pull these from ALL a player's past matches (both when they were the winner and the loser). You need to track stats per player_id regardless of winner/loser column.

**Features (6 total):** each expressed as `player1_stat - player2_stat`

### 3f. Match context features (direct from row)
- `surface` — one-hot encode: Hard, Clay, Grass, Carpet
- `tourney_level` — one-hot encode: G, M, A
- `round` — ordinal encode: R128=1, R64=2, R32=3, R16=4, QF=5, SF=6, F=7
- `best_of` — 3 or 5 (binary)
- `rank_diff` — player1_rank - player2_rank
- `rank_points_diff` — player1_rank_points - player2_rank_points
- `age_diff` — player1_age - player2_age
- `height_diff` — player1_height - player2_height
- `handedness` — encode matchup: RR=0, RL=1, LR=2, LL=3

---

## Stage 4 — Player Style Clustering (K-Means)

**Goal:** Assign each player a style archetype based on their career-average serve/return profile. This becomes a feature in the classifier.

### Steps:
1. Compute career-average serve stats per player across all their matches (2000–2025):
   - avg_ace_rate, avg_df_rate, avg_1st_serve_pct, avg_1st_serve_win_pct, avg_2nd_serve_win_pct, avg_bp_saved_pct
2. Standardize features (StandardScaler) — K-Means is sensitive to scale
3. Run K-Means for K = 2 through 8
4. Plot inertia vs K (elbow method) and select optimal K
5. Assign each player a cluster label (integer)

**Expected clusters (approximately):**
- Big servers: high ace rate, high 1st serve win %, dominant on grass
- All-court players: balanced profile
- Grinders/baseliners: low ace rate, high bp saved %, dominant on clay

### Features added to model:
- `player1_cluster` — one-hot encoded
- `player2_cluster` — one-hot encoded

---

## Stage 5 — Modeling

Train all three models on the same feature set. Use the validation set for hyperparameter tuning. Evaluate final performance on the test set only once.

### Full feature list for model input:
- `elo_diff`
- `global_form_diff`
- `surface_form_diff`
- `fatigue_diff`
- `avg_ace_rate_diff`
- `avg_df_rate_diff`
- `avg_1st_serve_pct_diff`
- `avg_1st_serve_win_pct_diff`
- `avg_2nd_serve_win_pct_diff`
- `avg_bp_saved_pct_diff`
- `rank_diff`
- `rank_points_diff`
- `age_diff`
- `height_diff`
- `best_of`
- `round` (ordinal)
- `surface_*` (one-hot)
- `tourney_level_*` (one-hot)
- `handedness`
- `player1_cluster_*` (one-hot)
- `player2_cluster_*` (one-hot)

### Model 1: Logistic Regression (baseline)
- Scale all features with StandardScaler before fitting
- Tune: `C` (regularization strength) on validation set
- Report: accuracy, log-loss, AUC-ROC on validation and test

### Model 2: Random Forest
- No scaling needed
- Tune: `n_estimators`, `max_depth`, `min_samples_leaf` on validation set
- Report: accuracy, log-loss, AUC-ROC
- Extract and plot feature importances

### Model 3: XGBoost (final model)
- No scaling needed
- Handles remaining nulls in serve stats natively
- Tune: `n_estimators`, `learning_rate`, `max_depth`, `subsample` on validation set
- Use early stopping on validation log-loss
- Report: accuracy, log-loss, AUC-ROC on test set

---

## Stage 6 — Evaluation & Insights

### Primary metrics on test set (2023–2025):
- **Log-loss** (primary) — lower is better
- **Accuracy** — % of matches correctly predicted
- **AUC-ROC** — model's ability to separate winners from losers

### Surface-stratified evaluation:
Report accuracy and log-loss separately for Hard, Clay, and Grass on the test set. This validates whether surface-specific features (Elo, form) are doing their job.

### Visualizations:
- Feature importance bar chart (XGBoost)
- Calibration curve — predicted probability vs actual win rate
- Confusion matrix on test set
- Accuracy by tourney_level (Grand Slams vs Masters vs 500s)
- Predicted upset probability distribution for notable upsets in 2023–2025

### Unsuccessful trials section (required by project doc):
- Neural Network (MLP): considered but excluded due to inferior performance on tabular data vs XGBoost and longer training time
- Note any features that hurt validation performance and were dropped

---

## Deliverable Structure

```
project/
│
├── data/
│   └── raw/           ← downloaded CSVs from Sackmann's repo
│
├── src/
│   ├── 01_ingestion.py        ← PySpark: load, union, filter
│   ├── 02_preprocessing.py    ← PySpark: derive stats, row flip, null handling
│   ├── 03_feature_engineering.py  ← Elo, form, fatigue, serve stats
│   ├── 04_clustering.py       ← K-Means player style clustering
│   ├── 05_modeling.py         ← LR, RF, XGBoost training & evaluation
│   └── 06_evaluation.py       ← metrics, plots, surface breakdown
│
└── outputs/
    ├── features.parquet       ← final feature matrix
    ├── models/                ← saved model files
    └── plots/                 ← all visualizations
```

---

## Citation
Jeff Sackmann / Tennis Abstract, tennis_atp dataset.
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
`https://github.com/JeffSackmann/tennis_atp`

Tennis Elo methodology: Jeff Sackmann, "An Introduction to Tennis Elo", Tennis Abstract, December 2019.
`https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/`
