#!/usr/bin/env python3
"""Bulk download TLE history from Space-Track for foundation model training.

Downloads the complete TLE history for all LEO objects (or a filtered set)
and saves as compressed numpy arrays ready for ML training.

Rate limit: 30 req/min, 300 req/hour. This script respects limits with
automatic pacing and resume capability.

Usage:
    export SPACETRACK_USER='...' SPACETRACK_PASS='...'
    python scripts/spacetrack_bulk_download.py --output data/spacetrack/

    # Resume after interruption (skips already-downloaded objects):
    python scripts/spacetrack_bulk_download.py --output data/spacetrack/ --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

BASE = "https://www.space-track.org"
LOGIN_URL = f"{BASE}/ajaxauth/login"

# Rate limits per Space-Track docs: 30 req/min AND 300 req/hour.
# Hourly is the binding constraint — 5 req/min = exactly 300 req/hour.
# Paced at 12.5 s between requests to stay comfortably under both.
REQUESTS_PER_HOUR = 280  # 20 under the 300/hr limit
DELAY_BETWEEN_REQUESTS = 3600.0 / REQUESTS_PER_HOUR

# Backoff after hitting rate limit signal
RATE_LIMIT_BACKOFF_S = 360  # 6 minutes — enough to clear the hourly bucket
MAX_RETRIES_PER_OBJECT = 3


def authenticate(client: httpx.Client, user: str, pw: str) -> bool:
    resp = client.post(LOGIN_URL, data={"identity": user, "password": pw})
    return resp.status_code == 200


def get_leo_catalog(client: httpx.Client, group: str = "") -> list[int]:
    """Get NORAD IDs of LEO objects (period < 128 min).

    Args:
        group: filter by object name. "starlink" = Starlink only.
               Empty = all LEO.
    """
    if group:
        url = (
            f"{BASE}/basicspacedata/query/class/gp"
            f"/OBJECT_NAME/~~{group}"
            f"/PERIOD/<128/EPOCH/>now-30"
            f"/orderby/NORAD_CAT_ID asc"
            f"/format/json"
        )
    else:
        url = (
            f"{BASE}/basicspacedata/query/class/gp"
            f"/PERIOD/<128/EPOCH/>now-30"
            f"/orderby/NORAD_CAT_ID asc"
            f"/format/json"
        )
    resp = client.get(url)
    if resp.status_code != 200:
        print(f"Catalog fetch failed: {resp.status_code}", file=sys.stderr)
        return []
    data = resp.json()
    return [int(r["NORAD_CAT_ID"]) for r in data if r.get("NORAD_CAT_ID")]


def _is_rate_limit_payload(records) -> bool:
    """Space-Track returns HTTP 200 with a single-element error array when
    rate-limited. Detect it so we don't save error responses as data."""
    if not isinstance(records, list) or len(records) != 1:
        return False
    first = records[0]
    if not isinstance(first, dict):
        return False
    err = first.get("error", "")
    return "rate limit" in err.lower() or "Acceptable Use" in err


def download_history(
    client: httpx.Client,
    norad_id: int,
    output_dir: Path,
) -> int:
    """Download full GP history for one NORAD ID. Returns record count.
    Returns -1 for resume-skip, 0 for no data, >0 for saved records.
    On rate limit, sleeps RATE_LIMIT_BACKOFF_S and retries up to
    MAX_RETRIES_PER_OBJECT times."""
    outfile = output_dir / f"{norad_id}.json.gz"
    if outfile.exists():
        return -1

    url = (
        f"{BASE}/basicspacedata/query/class/gp_history"
        f"/NORAD_CAT_ID/{norad_id}"
        f"/orderby/EPOCH asc"
        f"/format/json"
    )

    for attempt in range(1, MAX_RETRIES_PER_OBJECT + 1):
        resp = client.get(url)
        if resp.status_code == 429:
            print(
                f"  NORAD {norad_id}: HTTP 429, sleeping {RATE_LIMIT_BACKOFF_S}s "
                f"(attempt {attempt}/{MAX_RETRIES_PER_OBJECT})",
                file=sys.stderr,
            )
            time.sleep(RATE_LIMIT_BACKOFF_S)
            continue
        if resp.status_code != 200:
            print(f"  NORAD {norad_id}: HTTP {resp.status_code}", file=sys.stderr)
            return 0

        records = resp.json()

        # Body-level rate-limit (Space-Track returns 200 with error payload)
        if _is_rate_limit_payload(records):
            print(
                f"  NORAD {norad_id}: rate limit in body, sleeping "
                f"{RATE_LIMIT_BACKOFF_S}s (attempt {attempt}/{MAX_RETRIES_PER_OBJECT})",
                file=sys.stderr,
            )
            time.sleep(RATE_LIMIT_BACKOFF_S)
            continue

        if not records:
            return 0

        import gzip
        with gzip.open(outfile, "wt") as f:
            json.dump(records, f)
        return len(records)

    print(
        f"  NORAD {norad_id}: gave up after {MAX_RETRIES_PER_OBJECT} retries",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk download Space-Track TLE history for ML training.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/spacetrack"),
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-downloaded NORAD IDs")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max objects to download (0 = all)")
    parser.add_argument("--min-norad", type=int, default=0,
                        help="Start from this NORAD ID")
    parser.add_argument("--group", type=str, default="",
                        help="Filter by name (e.g. 'starlink'). Empty = all LEO.")
    args = parser.parse_args(argv)

    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    if not user or not pw:
        print("Set SPACETRACK_USER and SPACETRACK_PASS", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        print("Authenticating...")
        if not authenticate(client, user, pw):
            print("Auth failed", file=sys.stderr)
            return 1

        group_label = f" (group={args.group})" if args.group else ""
        print(f"Fetching LEO catalog{group_label}...")
        catalog = get_leo_catalog(client, group=args.group)
        print(f"LEO objects: {len(catalog)}")

        if args.min_norad > 0:
            catalog = [n for n in catalog if n >= args.min_norad]
            print(f"  filtered to {len(catalog)} (min NORAD {args.min_norad})")

        if args.limit > 0:
            catalog = catalog[:args.limit]
            print(f"  limited to {args.limit}")

        # Check resume
        if args.resume:
            existing = {int(f.stem.split(".")[0]) for f in args.output.glob("*.json.gz")}
            before = len(catalog)
            catalog = [n for n in catalog if n not in existing]
            print(f"  resume: {before - len(catalog)} already done, {len(catalog)} remaining")

        total_records = 0
        t0 = time.time()

        for i, norad_id in enumerate(catalog):
            n = download_history(client, norad_id, args.output)

            if n > 0:
                total_records += n
            elif n == -1:
                continue  # skipped

            # Progress
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            eta_h = (len(catalog) - i - 1) / rate if rate > 0 else 0

            if (i + 1) % 50 == 0 or i == len(catalog) - 1:
                print(
                    f"  [{i+1}/{len(catalog)}] NORAD {norad_id}: {n} records | "
                    f"total {total_records:,} | {rate:.0f} obj/h | ETA {eta_h:.1f}h"
                )

            # Rate limit
            time.sleep(DELAY_BETWEEN_REQUESTS)

    elapsed_h = (time.time() - t0) / 3600
    print(f"\nDone: {total_records:,} records from {len(catalog)} objects in {elapsed_h:.1f}h")
    print(f"Saved to {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
