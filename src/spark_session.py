"""Helper for building a pseudo-distributed Spark session."""
from __future__ import annotations

import os
import sys
from pyspark.sql import SparkSession


def get_spark(app_name: str = "atp_tennis", master: str = "local[*]") -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
