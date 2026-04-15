#!/usr/bin/env python3
"""One-shot backfill of tle.bstar for pre-v4 rows.

Schema v4 added the bstar column to tle, but upsert_tles is INSERT OR
IGNORE on (norad_id, epoch_jd) — legacy rows never get updated because
their keys already exist and fresh fetches are silently dropped.

Fix: walk every row where bstar IS NULL, parse bstar from the stored
line1 text, and UPDATE in place. Batched in 5k-row transactions so a
mid-run kill doesn't lose progress, and safe to re-run (only touches
NULL rows).

The parse helper is inlined so the script is self-contained — no
services.* imports — and can run on any Python 3 with stdlib sqlite3,
including directly on the VPS host without going through docker.

Usage:
    python3 scripts/backfill_bstar.py [db_path]
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

BATCH_SIZE = 5000


def _parse_tle_float(s: str) -> float:
    """TLE's compact float format: signed mantissa with implied decimal
    point + trailing [+-]d exponent. Inlined from services.telemetry.tle_fetcher
    to keep this script self-contained.
    """
    s = s.strip()
    if not s:
        return 0.0
    if len(s) >= 2 and s[-2] in "+-" and s[-1].isdigit():
        exp = int(s[-2:])
        mantissa_str = s[:-2]
    else:
        return float(s)
    if not mantissa_str:
        return 0.0
    if mantissa_str[0] in "+-":
        sign = mantissa_str[0]
        digits = mantissa_str[1:]
        mantissa = float(f"{sign}0.{digits}") if digits else 0.0
    else:
        mantissa = float(f"0.{mantissa_str}")
    return mantissa * (10 ** exp)


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
