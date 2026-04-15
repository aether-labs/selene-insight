#!/usr/bin/env python3
"""One-shot backfill of tle.bstar for pre-v4 rows.

Schema v4 added the bstar column to tle, but upsert_tles is INSERT OR
IGNORE on (norad_id, epoch_jd) — legacy rows stay with bstar=NULL
forever because their (norad_id, epoch_jd) keys already exist and new
fetches are silently dropped.

Fix: walk every row where bstar IS NULL, parse bstar from the stored
line1 text using the same helper the fetcher uses, and UPDATE in place.
Batched commits so a kill mid-run doesn't lose progress.

Usage:
    python scripts/backfill_bstar.py [db_path]

Safe to run multiple times — only touches rows where bstar IS NULL.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

# Make services.* importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.telemetry.tle_fetcher import _parse_tle_float

BATCH_SIZE = 5000


def main(db_path: str) -> int:
    if not os.path.exists(db_path):
        print(f"error: {db_path} does not exist", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    total = conn.execute(
        "SELECT COUNT(*) FROM tle WHERE bstar IS NULL"
    ).fetchone()[0]
    print(f"[backfill] {total} rows to update")

    if total == 0:
        conn.close()
        return 0

    t0 = time.time()
    updated = 0
    failures = 0

    while True:
        rows = conn.execute(
            "SELECT id, line1 FROM tle WHERE bstar IS NULL LIMIT ?",
            (BATCH_SIZE,),
        ).fetchall()
        if not rows:
            break

        updates: list[tuple[float | None, int]] = []
        for row_id, line1 in rows:
            try:
                if line1 and len(line1) >= 61:
                    bstar = _parse_tle_float(line1[53:61])
                else:
                    bstar = None
                    failures += 1
            except (ValueError, IndexError):
                bstar = None
                failures += 1
            updates.append((bstar, row_id))

        conn.executemany("UPDATE tle SET bstar = ? WHERE id = ?", updates)
        conn.commit()
        updated += len(updates)
        elapsed = time.time() - t0
        rate = updated / elapsed if elapsed > 0 else 0
        print(
            f"[backfill] {updated}/{total} ({100*updated/total:.1f}%) "
            f"at {rate:.0f} rows/s"
        )

    conn.close()
    print(f"[backfill] done: {updated} updated, {failures} failed")
    return 0


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "ARGUS_DB_PATH", "data/starlink.db"
    )
    sys.exit(main(db))
