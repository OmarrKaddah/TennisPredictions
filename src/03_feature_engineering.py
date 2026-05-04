"""Stage 3 - Feature Engineering.

Reads preprocessed matches, walks chronologically to build per-match
features that depend on a player's prior history (Elo, rolling form,
fatigue, rolling serve stats), then performs the seeded winner/loser
row flip and emits a player1/player2 feature matrix.

Sequential computations (Elo, rolling windows) run in pandas on the
driver - the cleaned dataset is ~70k rows which fits comfortably in
memory. We use Spark only to load the preprocessed parquet and to
write the final feature parquet.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    ELO_INITIAL,
    ELO_K,
    FATIGUE_DAYS,
    FATIGUE_GRAND_SLAM_WEIGHT,
    FEATURES_PARQUET,
    GLOBAL_FORM_WINDOW,
    MAIN_DRAW_LEVELS,
    PREPROCESSED_PARQUET,
    RANDOM_SEED,
    RANKINGS_PARQUET,
    SERVE_STATS_WINDOW,
    SURFACE_FORM_WINDOW,
)
from spark_session import get_spark


SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SERVE_FIELDS = (
    "ace_rate",
    "df_rate",
    "1st_serve_pct",
    "1st_serve_win_pct",
    "2nd_serve_win_pct",
    "bp_saved_pct",
)
ROUND_ORDINAL = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 4, "BR": 5}
HAND_CODE = {("R", "R"): 0, ("R", "L"): 1, ("L", "R"): 2, ("L", "L"): 3}


def load_preprocessed() -> pd.DataFrame:
    spark = get_spark("atp_features_load")
    spark.sparkContext.setLogLevel("WARN")
    pdf = spark.read.parquet(str(PREPROCESSED_PARQUET)).toPandas()
    spark.stop()
    sort_cols = ["tourney_date"] + (["match_num"] if "match_num" in pdf.columns else [])
    pdf = pdf.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return pdf


def build_peak_rank_lookup() -> Dict[int, tuple]:
    """Per-player (sorted_dates, cumulative_min_rank) arrays.

    For a query date d, the peak rank achieved on or before d is
    `peaks[searchsorted(dates, d, side='right') - 1]`. Returns an
    empty dict if rankings.parquet is unavailable.
    """
    if not RANKINGS_PARQUET.exists():
        print("[warn] rankings.parquet missing - peak_rank features will be NaN")
        return {}
    if RANKINGS_PARQUET.is_dir():
        files = sorted(Path(RANKINGS_PARQUET).glob("*.parquet"))
        rk = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        rk = pd.read_parquet(RANKINGS_PARQUET)
    rk = rk[["player_id", "ranking_date", "rank"]].dropna()
    rk["player_id"] = rk["player_id"].astype(int)
    rk["ranking_date"] = rk["ranking_date"].astype(int)
    rk["rank"] = rk["rank"].astype(int)
    rk = rk.sort_values(["player_id", "ranking_date"], kind="mergesort")
    rk["peak_to_date"] = rk.groupby("player_id")["rank"].cummin()

    lookup: Dict[int, tuple] = {}
    for pid, sub in rk.groupby("player_id", sort=False):
        lookup[int(pid)] = (
            sub["ranking_date"].to_numpy(),
            sub["peak_to_date"].to_numpy(),
        )
    print(f"Peak-rank lookup built for {len(lookup):,} players")
    return lookup


def peak_rank_for(lookup: Dict[int, tuple], pid: int, tdate: int) -> float:
    pair = lookup.get(pid)
    if pair is None:
        return float("nan")
    dates, peaks = pair
    idx = int(np.searchsorted(dates, tdate, side="right")) - 1
    if idx < 0:
        return float("nan")
    return float(peaks[idx])


def expected_score(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))


def deque_mean(d: Deque[float]) -> float:
    if not d:
        return np.nan
    return float(np.mean(d))


def serve_stats_mean(history: Deque[Dict[str, float]], field: str) -> float:
    if not history:
        return np.nan
    vals = [m[field] for m in history if not np.isnan(m[field])]
    return float(np.mean(vals)) if vals else np.nan


def compute_features(
    matches: pd.DataFrame, peak_lookup: Dict[int, tuple] | None = None
) -> pd.DataFrame:
    """Walk matches chronologically and emit pre-match features per row."""
    if peak_lookup is None:
        peak_lookup = {}
    elo: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {s: ELO_INITIAL for s in SURFACES}
    )
    global_form: Dict[int, Deque[float]] = defaultdict(
        lambda: deque(maxlen=GLOBAL_FORM_WINDOW)
    )
    surface_form: Dict[int, Dict[str, Deque[float]]] = defaultdict(
        lambda: {s: deque(maxlen=SURFACE_FORM_WINDOW) for s in SURFACES}
    )
    recent_matches: Dict[int, Deque[tuple]] = defaultdict(deque)
    serve_history: Dict[int, Deque[Dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=SERVE_STATS_WINDOW)
    )

    rows = []
    n = len(matches)
    print(f"Processing {n:,} matches chronologically...")

    for idx, m in enumerate(matches.itertuples(index=False)):
        if idx % 10000 == 0 and idx > 0:
            print(f"  ...{idx:,}/{n:,}")

        surface = m.surface if m.surface in SURFACES else "Hard"
        tdate = int(m.tourney_date)
        wid = int(m.winner_id)
        lid = int(m.loser_id)
        level = m.tourney_level

        w_elo = elo[wid][surface]
        l_elo = elo[lid][surface]

        w_global = deque_mean(global_form[wid])
        l_global = deque_mean(global_form[lid])

        w_surf = deque_mean(surface_form[wid][surface])
        l_surf = deque_mean(surface_form[lid][surface])

        cutoff = _shift_yyyymmdd(tdate, -FATIGUE_DAYS)
        _prune_recent(recent_matches[wid], cutoff)
        _prune_recent(recent_matches[lid], cutoff)
        w_fat = sum(
            FATIGUE_GRAND_SLAM_WEIGHT if lvl == "G" else 1.0
            for _, lvl in recent_matches[wid]
        )
        l_fat = sum(
            FATIGUE_GRAND_SLAM_WEIGHT if lvl == "G" else 1.0
            for _, lvl in recent_matches[lid]
        )

        w_serve = {f: serve_stats_mean(serve_history[wid], f) for f in SERVE_FIELDS}
        l_serve = {f: serve_stats_mean(serve_history[lid], f) for f in SERVE_FIELDS}

        w_peak = peak_rank_for(peak_lookup, wid, tdate)
        l_peak = peak_rank_for(peak_lookup, lid, tdate)

        row = {
            "tourney_date": tdate,
            "match_num": getattr(m, "match_num", None),
            "tourney_level": level,
            "surface": surface,
            "round": getattr(m, "round", None),
            "best_of": getattr(m, "best_of", None),
            "match_source": getattr(m, "match_source", "main"),
            "w_id": wid,
            "l_id": lid,
            "w_name": getattr(m, "winner_name", None),
            "l_name": getattr(m, "loser_name", None),
            "w_elo": w_elo,
            "l_elo": l_elo,
            "w_global_form": w_global,
            "l_global_form": l_global,
            "w_surface_form": w_surf,
            "l_surface_form": l_surf,
            "w_fatigue": w_fat,
            "l_fatigue": l_fat,
            "w_rank": getattr(m, "winner_rank", None),
            "l_rank": getattr(m, "loser_rank", None),
            "w_rank_points": getattr(m, "winner_rank_points", None),
            "l_rank_points": getattr(m, "loser_rank_points", None),
            "w_age": getattr(m, "winner_age", None),
            "l_age": getattr(m, "loser_age", None),
            "w_ht": getattr(m, "winner_ht", None),
            "l_ht": getattr(m, "loser_ht", None),
            "w_hand": getattr(m, "winner_hand", None) or "U",
            "l_hand": getattr(m, "loser_hand", None) or "U",
            "w_peak_rank": w_peak,
            "l_peak_rank": l_peak,
        }
        for f in SERVE_FIELDS:
            row[f"w_avg_{f}"] = w_serve[f]
            row[f"l_avg_{f}"] = l_serve[f]
        rows.append(row)

        e_w = expected_score(w_elo, l_elo)
        e_l = 1.0 - e_w
        elo[wid][surface] = w_elo + ELO_K * (1.0 - e_w)
        elo[lid][surface] = l_elo + ELO_K * (0.0 - e_l)

        global_form[wid].append(1.0)
        global_form[lid].append(0.0)
        surface_form[wid][surface].append(1.0)
        surface_form[lid][surface].append(0.0)
        recent_matches[wid].append((tdate, level))
        recent_matches[lid].append((tdate, level))

        w_match_serve = _serve_dict_from_row(m, "w")
        l_match_serve = _serve_dict_from_row(m, "l")
        serve_history[wid].append(w_match_serve)
        serve_history[lid].append(l_match_serve)

    return pd.DataFrame(rows)


def _serve_dict_from_row(m, side: str) -> Dict[str, float]:
    return {
        "ace_rate": _safe_get(m, f"{side}_ace_rate"),
        "df_rate": _safe_get(m, f"{side}_df_rate"),
        "1st_serve_pct": _safe_get(m, f"{side}_1st_serve_pct"),
        "1st_serve_win_pct": _safe_get(m, f"{side}_1st_serve_win_pct"),
        "2nd_serve_win_pct": _safe_get(m, f"{side}_2nd_serve_win_pct"),
        "bp_saved_pct": _safe_get(m, f"{side}_bp_saved_pct"),
    }


def _safe_get(m, attr: str) -> float:
    v = getattr(m, attr, None)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _shift_yyyymmdd(yyyymmdd: int, days: int) -> int:
    d = pd.to_datetime(str(yyyymmdd), format="%Y%m%d") + pd.Timedelta(days=days)
    return int(d.strftime("%Y%m%d"))


def _prune_recent(dq: Deque[tuple], cutoff_yyyymmdd: int) -> None:
    while dq and dq[0][0] < cutoff_yyyymmdd:
        dq.popleft()


def flip_and_diff(features: pd.DataFrame) -> pd.DataFrame:
    """Apply seeded winner/loser flip and compute difference features."""
    rng = np.random.default_rng(RANDOM_SEED)
    flip = rng.random(len(features)) < 0.5

    pairs = [
        ("elo", "elo"),
        ("global_form", "global_form"),
        ("surface_form", "surface_form"),
        ("fatigue", "fatigue"),
        ("rank", "rank"),
        ("rank_points", "rank_points"),
        ("age", "age"),
        ("ht", "ht"),
        ("hand", "hand"),
        ("id", "id"),
        ("name", "name"),
        ("peak_rank", "peak_rank"),
    ]
    for f in SERVE_FIELDS:
        pairs.append((f"avg_{f}", f"avg_{f}"))

    out = pd.DataFrame(index=features.index)
    out["tourney_date"] = features["tourney_date"].astype(int)
    out["match_num"] = features.get("match_num")
    out["tourney_level"] = features["tourney_level"]
    out["surface"] = features["surface"]
    out["round"] = features["round"]
    out["best_of"] = features["best_of"]

    for canon, _ in pairs:
        w = features[f"w_{canon}"].to_numpy()
        l = features[f"l_{canon}"].to_numpy()
        p1 = np.where(flip, w, l)
        p2 = np.where(flip, l, w)
        out[f"player1_{canon}"] = p1
        out[f"player2_{canon}"] = p2

    out["target"] = np.where(flip, 1, 0).astype(np.int8)

    out["elo_diff"] = pd.to_numeric(out["player1_elo"], errors="coerce") - pd.to_numeric(out["player2_elo"], errors="coerce")
    out["global_form_diff"] = pd.to_numeric(out["player1_global_form"], errors="coerce") - pd.to_numeric(out["player2_global_form"], errors="coerce")
    out["surface_form_diff"] = pd.to_numeric(out["player1_surface_form"], errors="coerce") - pd.to_numeric(out["player2_surface_form"], errors="coerce")
    out["fatigue_diff"] = pd.to_numeric(out["player1_fatigue"], errors="coerce") - pd.to_numeric(out["player2_fatigue"], errors="coerce")
    out["rank_diff"] = pd.to_numeric(out["player1_rank"], errors="coerce") - pd.to_numeric(out["player2_rank"], errors="coerce")
    out["rank_points_diff"] = pd.to_numeric(out["player1_rank_points"], errors="coerce") - pd.to_numeric(out["player2_rank_points"], errors="coerce")
    out["age_diff"] = pd.to_numeric(out["player1_age"], errors="coerce") - pd.to_numeric(out["player2_age"], errors="coerce")
    out["height_diff"] = pd.to_numeric(out["player1_ht"], errors="coerce") - pd.to_numeric(out["player2_ht"], errors="coerce")
    out["peak_rank_diff"] = pd.to_numeric(out["player1_peak_rank"], errors="coerce") - pd.to_numeric(out["player2_peak_rank"], errors="coerce")

    for f in SERVE_FIELDS:
        out[f"avg_{f}_diff"] = pd.to_numeric(out[f"player1_avg_{f}"], errors="coerce") - pd.to_numeric(out[f"player2_avg_{f}"], errors="coerce")

    out["round_ordinal"] = out["round"].map(ROUND_ORDINAL).fillna(0).astype(int)

    p1h = out["player1_hand"].fillna("R").str[0].str.upper().replace({"U": "R"})
    p2h = out["player2_hand"].fillna("R").str[0].str.upper().replace({"U": "R"})
    out["handedness"] = [HAND_CODE.get((a, b), 0) for a, b in zip(p1h, p2h)]

    for s in SURFACES:
        out[f"surface_{s}"] = (out["surface"] == s).astype(int)
    for lvl in ("G", "M", "A"):
        out[f"tourney_level_{lvl}"] = (out["tourney_level"] == lvl).astype(int)

    out["best_of"] = pd.to_numeric(out["best_of"], errors="coerce").fillna(3).astype(int)
    return out


def main() -> int:
    if not PREPROCESSED_PARQUET.exists():
        print(f"[error] missing {PREPROCESSED_PARQUET}. Run: python src/02_preprocessing.py")
        return 2

    matches = load_preprocessed()
    print(f"Loaded preprocessed: {len(matches):,} rows")

    peak_lookup = build_peak_rank_lookup()

    features = compute_features(matches, peak_lookup)
    print(f"Per-match features computed (all sources): {features.shape}")

    main_mask = (
        (features["match_source"] == "main")
        & (features["tourney_level"].isin(MAIN_DRAW_LEVELS))
    )
    main_features = features[main_mask].reset_index(drop=True)
    print(
        f"Main-draw rows kept for prediction: {len(main_features):,} "
        f"(filtered out {len(features) - len(main_features):,} challenger/qualifying)"
    )

    final = flip_and_diff(main_features)
    print(f"Player1/player2 feature matrix: {final.shape}")

    out_path = str(FEATURES_PARQUET)
    final.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
