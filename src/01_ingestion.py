"""Stage 1 - Data Ingestion (PySpark).

Loads four datasets from Sackmann's repo:
  1. atp_matches_YYYY.csv          - main-draw ATP matches
  2. atp_matches_qual_chall_YYYY.csv - challenger + qualifying matches
  3. atp_rankings_*.csv            - weekly rankings (multi-decade)
  4. atp_players.csv               - player metadata

Main-draw and challenger/qualifier files share an identical 49-column
schema, so we union them and tag each row with `match_source` ('main'
or 'qual_chall'). Including challenger matches is what pushes the
processed dataset above the 100 MB course requirement and gives
better player Elo / rolling-form estimates (especially for younger
players who started on challenger). Stage 3 only emits prediction
rows for main-draw G/M/A matches; the rest are used purely to update
each player's running state.
"""
from __future__ import annotations

import sys
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_PARQUET,
    KEEP_LEVELS,
    PLAYERS_PARQUET,
    RANKINGS_PARQUET,
    RANKING_DECADES,
    RAW_DIR,
    YEARS,
)
from spark_session import get_spark


BAD_SCORE_TOKENS = ("RET", "W/O", "DEF", "ABN", "WO")


def load_match_csv(spark, path: Path, source_tag: str) -> DataFrame | None:
    if not path.exists():
        print(f"[warn] missing {path.name}")
        return None
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(str(path))
    )
    return df.withColumn("match_source", F.lit(source_tag))


def collect_match_dfs(spark) -> list[DataFrame]:
    dfs: list[DataFrame] = []
    for y in YEARS:
        for tag, prefix in (("main", "atp_matches"), ("qual_chall", "atp_matches_qual_chall")):
            df = load_match_csv(spark, RAW_DIR / f"{prefix}_{y}.csv", tag)
            if df is not None:
                dfs.append(df)
    return dfs


def filter_and_clean(matches: DataFrame) -> DataFrame:
    matches = matches.filter(F.col("tourney_level").isin(*KEEP_LEVELS))
    score_col = F.coalesce(F.col("score").cast("string"), F.lit(""))
    bad_score_filter = reduce(
        lambda acc, tok: acc | F.upper(score_col).contains(tok),
        BAD_SCORE_TOKENS,
        F.lit(False),
    )
    matches = matches.filter(~bad_score_filter)
    matches = matches.withColumn("tourney_date", F.col("tourney_date").cast("int"))
    if "match_num" in matches.columns:
        matches = matches.withColumn("match_num", F.col("match_num").cast("int"))
        matches = matches.orderBy(F.col("tourney_date").asc(), F.col("match_num").asc())
    else:
        matches = matches.orderBy(F.col("tourney_date").asc())
    return matches


def ingest_matches(spark) -> int:
    dfs = collect_match_dfs(spark)
    if not dfs:
        print("[error] no match CSVs found - run src/00_download_data.py")
        return 1

    common_cols = sorted(set.intersection(*(set(df.columns) for df in dfs)))
    dfs = [df.select(*common_cols) for df in dfs]
    print(f"Match files loaded: {len(dfs)}, common columns: {len(common_cols)}")

    matches = reduce(DataFrame.unionByName, dfs)
    matches = filter_and_clean(matches).cache()

    n_total = matches.count()
    by_source = matches.groupBy("match_source").count().collect()
    print(f"Total cleaned match rows: {n_total:,}")
    for r in by_source:
        print(f"  match_source={r['match_source']}: {r['count']:,}")

    matches.coalesce(1).write.mode("overwrite").parquet(str(CLEANED_PARQUET))
    print(f"Wrote {CLEANED_PARQUET}")
    return 0


def ingest_rankings(spark) -> None:
    paths = [RAW_DIR / f"atp_rankings_{d}.csv" for d in RANKING_DECADES]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("[warn] no rankings CSVs found - skipping rankings ingest")
        return
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv([str(p) for p in paths])
    )
    df = df.withColumnRenamed("player", "player_id")
    df = df.withColumn("ranking_date", F.col("ranking_date").cast("int"))
    df = df.withColumn("rank", F.col("rank").cast("int"))
    df = df.withColumn("points", F.col("points").cast("int"))
    n = df.count()
    print(f"Rankings rows: {n:,}")
    df.coalesce(1).write.mode("overwrite").parquet(str(RANKINGS_PARQUET))
    print(f"Wrote {RANKINGS_PARQUET}")


def ingest_players(spark) -> None:
    path = RAW_DIR / "atp_players.csv"
    if not path.exists():
        print("[warn] atp_players.csv missing - skipping players ingest")
        return
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )
    n = df.count()
    print(f"Player rows: {n:,}")
    df.coalesce(1).write.mode("overwrite").parquet(str(PLAYERS_PARQUET))
    print(f"Wrote {PLAYERS_PARQUET}")


def main() -> int:
    if not any(RAW_DIR.glob("atp_matches_*.csv")):
        print(f"[error] no CSVs in {RAW_DIR}. Run: python src/00_download_data.py")
        return 2

    spark = get_spark("atp_ingestion")
    spark.sparkContext.setLogLevel("WARN")

    rc = ingest_matches(spark)
    if rc != 0:
        spark.stop()
        return rc
    ingest_rankings(spark)
    ingest_players(spark)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
