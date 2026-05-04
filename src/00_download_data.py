"""Download Sackmann's ATP datasets used by the pipeline.

Targets:
- atp_matches_YYYY.csv         (main-draw matches, 2000-2025)
- atp_matches_qual_chall_YYYY.csv  (challenger + qualifying, 2000-2025)
- atp_rankings_{90s,00s,10s,20s,current}.csv  (weekly rankings)
- atp_players.csv              (player metadata)

Skips files that already exist. Treats HTTP 404 as 'not yet published'
(soft skip) so missing future-year files don't fail the pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent))
from config import RANKING_DECADES, RAW_DIR, SACKMANN_BASE, YEARS


NOT_PUBLISHED = "not_published"
NETWORK_FAIL = "network_fail"
OK = "ok"


def _download(url: str, target: Path, retries: int = 3, sleep_s: float = 1.0) -> str:
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target.name} ({target.stat().st_size:,} bytes)")
        return OK
    last_status = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[get ] {url} (attempt {attempt})")
            r = requests.get(url, timeout=60)
            last_status = r.status_code
            if r.status_code == 200 and len(r.content) > 0:
                target.write_bytes(r.content)
                print(f"[ok  ] saved {target.name} ({len(r.content):,} bytes)")
                return OK
            if r.status_code == 404:
                print(f"[miss] {target.name} not yet published (HTTP 404)")
                return NOT_PUBLISHED
            print(f"[warn] HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"[err ] {e}")
        time.sleep(sleep_s)
    return NOT_PUBLISHED if last_status == 404 else NETWORK_FAIL


def download_all() -> tuple[list, list]:
    not_published: list[str] = []
    failures: list[str] = []

    targets: list[tuple[str, Path]] = []
    for y in YEARS:
        targets.append(
            (f"{SACKMANN_BASE}/atp_matches_{y}.csv", RAW_DIR / f"atp_matches_{y}.csv")
        )
        targets.append(
            (
                f"{SACKMANN_BASE}/atp_matches_qual_chall_{y}.csv",
                RAW_DIR / f"atp_matches_qual_chall_{y}.csv",
            )
        )
    for d in RANKING_DECADES:
        targets.append(
            (f"{SACKMANN_BASE}/atp_rankings_{d}.csv", RAW_DIR / f"atp_rankings_{d}.csv")
        )
    targets.append((f"{SACKMANN_BASE}/atp_players.csv", RAW_DIR / "atp_players.csv"))

    for url, dest in targets:
        result = _download(url, dest)
        if result == NOT_PUBLISHED:
            not_published.append(dest.name)
        elif result == NETWORK_FAIL:
            failures.append(dest.name)
    return not_published, failures


def main() -> int:
    not_published, failures = download_all()
    if not_published:
        print(f"\nNot yet published (skipped): {len(not_published)} files")
        for n in not_published:
            print(f"  - {n}")
    if failures:
        print(f"\nNetwork/transient failures (retry later):")
        for n in failures:
            print(f"  - {n}")
        return 1

    files = sorted(RAW_DIR.glob("*.csv"))
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} CSVs in {RAW_DIR}, total {total_bytes/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
