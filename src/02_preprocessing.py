"""Stage 2 - Preprocessing (PySpark).

Derives serve percentage columns and applies the documented null-handling
rules. The chronological row flip and difference features are deferred
to feature engineering, since rolling stats must be computed on the
original winner/loser layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLEANED_PARQUET, PREPROCESSED_PARQUET
from spark_session import get_spark


SIDES = ("w", "l")


def safe_div(num, den):
    return F.when(den > 0, num / den).otherwise(F.lit(None).cast("double"))


def derive_serve_columns(df: DataFrame) -> DataFrame:
    for s in SIDES:
        svpt = F.col(f"{s}_svpt")
        first_in = F.col(f"{s}_1stIn")
        first_won = F.col(f"{s}_1stWon")
        second_won = F.col(f"{s}_2ndWon")
        bp_saved = F.col(f"{s}_bpSaved")
        bp_faced = F.col(f"{s}_bpFaced")
        ace = F.col(f"{s}_ace")
        df_col = F.col(f"{s}_df")

        df = (
            df.withColumn(f"{s}_1st_serve_pct", safe_div(first_in, svpt))
            .withColumn(f"{s}_1st_serve_win_pct", safe_div(first_won, first_in))
            .withColumn(f"{s}_2nd_serve_win_pct", safe_div(second_won, svpt - first_in))
            .withColumn(f"{s}_bp_saved_pct", safe_div(bp_saved, bp_faced))
            .withColumn(f"{s}_ace_rate", safe_div(ace, svpt))
            .withColumn(f"{s}_df_rate", safe_div(df_col, svpt))
        )
    return df


def fill_basic_nulls(df: DataFrame) -> DataFrame:
    for s in SIDES:
        rank_col = f"{s}_rank"
        df = df.withColumn(rank_col, F.coalesce(F.col(rank_col), F.lit(999)).cast("int"))

    height_median_per_surface = (
        df.select("surface", "winner_ht", "loser_ht")
        .selectExpr("surface", "stack(2, winner_ht, loser_ht) as ht")
        .filter(F.col("ht").isNotNull())
        .groupBy("surface")
        .agg(F.expr("percentile_approx(ht, 0.5)").alias("ht_median"))
    )
    df = df.join(height_median_per_surface, on="surface", how="left")
    for s in SIDES:
        df = df.withColumn(
            f"{s}_ht",
            F.coalesce(F.col(f"{s}_ht"), F.col("ht_median")).cast("double"),
        )
    df = df.drop("ht_median")
    return df


def main() -> int:
    if not CLEANED_PARQUET.exists():
        print(f"[error] missing {CLEANED_PARQUET}. Run: python src/01_ingestion.py")
        return 2

    spark = get_spark("atp_preprocessing")
    spark.sparkContext.setLogLevel("WARN")

    src = str(CLEANED_PARQUET)
    print(f"Reading {src}")
    df = spark.read.parquet(src)

    df = derive_serve_columns(df)
    df = fill_basic_nulls(df)

    sort_cols = [F.col("tourney_date").asc()]
    if "match_num" in df.columns:
        sort_cols.append(F.col("match_num").asc())
    df = df.orderBy(*sort_cols)

    out = str(PREPROCESSED_PARQUET)
    df.coalesce(1).write.mode("overwrite").parquet(out)
    print(f"Rows: {df.count():,}")
    print(f"Wrote {out}")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
