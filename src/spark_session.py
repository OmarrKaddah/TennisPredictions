"""Helper for building a pseudo-distributed Spark session."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession


def _configure_hadoop_home_windows() -> str | None:
    """On Windows, Spark needs HADOOP_HOME pointing at a dir that contains
    bin/winutils.exe and bin/hadoop.dll. Returns the path used, or None if
    not on Windows."""
    if sys.platform != "win32":
        return None
    hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
    Path(hadoop_home, "bin").mkdir(parents=True, exist_ok=True)
    os.environ["HADOOP_HOME"] = hadoop_home
    return hadoop_home


def get_spark(app_name: str = "atp_tennis", master: str = "local[*]") -> SparkSession:
    # Use "python" (resolved via PATH) rather than sys.executable — avoids
    # PySpark failing when the venv path contains spaces (Windows OneDrive paths).
    os.environ.setdefault("PYSPARK_PYTHON", "python")
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python")
    hadoop_home = _configure_hadoop_home_windows()

    java_opts = "-Dfile.encoding=UTF-8"
    if hadoop_home:
        # Java requires forward slashes even on Windows
        hadoop_home_fwd = hadoop_home.replace("\\", "/")
        java_opts += f" -Dhadoop.home.dir={hadoop_home_fwd}"

    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.driver.extraJavaOptions", java_opts)
        .getOrCreate()
    )
