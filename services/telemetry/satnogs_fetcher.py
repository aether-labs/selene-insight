"""SatNOGS observation fetcher — RF signal detection for gap cross-validation.

Ingests satellite observation results from the SatNOGS network (~300 ground
stations worldwide). The key signal is NOT position accuracy — it's
**existence detection**: did a ground station hear the satellite's RF
transmission during a predicted pass?

For gap cross-validation:
  TLE gap (no new TLE >24h) + RF silence (SatNOGS observations fail)
  = high-confidence satellite failure / breakup signal

SatNOGS API docs: https://network.satnogs.org/api/
No authentication required. Rate limit: be polite (~1 req/s).
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

try:
    from services.telemetry.store import StarlinkStore
except ImportError:
    from store import StarlinkStore  # type: ignore

SATNOGS_API = "https://network.satnogs.org/api"
PAGE_SIZE = 100
FETCH_INTERVAL = 6 * 3600  # every 6h (offset from TLE fetcher's 8h)
# How far back to look on each fetch (overlapping window ensures no gaps)
LOOKBACK_HOURS = 12


async def _fetch_page(client: httpx.AsyncClient, url: str) -> tuple[list, str | None]:
    """Fetch one page. Returns (results, next_url)."""
    resp = await client.get(url)
    resp.raise_for_status()
    data = resp.json()
    # SatNOGS paginates via Link header or next field
    next_url = None
    if isinstance(data, dict) and "next" in data:
        next_url = data.get("next")
        data = data.get("results", [])
    return data, next_url


async def fetch_observations(
    norad_ids: list[int],
    since: datetime,
    until: datetime | None = None,
) -> list[dict]:
    """Fetch SatNOGS observations for a set of NORAD IDs in a time window.

    Handles pagination. Returns raw observation dicts.
    """
    results: list[dict] = []
    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "argusorb/0.2 (satnogs-fetcher)"},
    ) as client:
        for norad_id in norad_ids:
            params = {
                "satellite__norad_cat_id": norad_id,
                "start": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "page_size": PAGE_SIZE,
                "format": "json",
            }
            if until:
                params["end"] = until.strftime("%Y-%m-%dT%H:%M:%SZ")

            url = f"{SATNOGS_API}/observations/"
            page = 0
            while url:
                try:
                    resp = await client.get(url, params=params if page == 0 else None)
                    resp.raise_for_status()
                    data = resp.json()

                    # Handle both paginated and flat responses
                    if isinstance(data, list):
                        page_results = data
                        url = None
                    elif isinstance(data, dict):
                        page_results = data.get("results", [])
                        url = data.get("next")
                    else:
                        break

                    for obs in page_results:
                        obs["_norad_id"] = norad_id
                    results.extend(page_results)
                    page += 1

                    if not page_results:
                        break
                    # Rate limit politeness
                    await asyncio.sleep(0.5)
                except httpx.HTTPStatusError as e:
                    print(
                        f"[SATNOGS] HTTP {e.response.status_code} for NORAD {norad_id}",
                        file=sys.stderr,
                    )
                    break
                except Exception as e:
                    print(
                        f"[SATNOGS] Error for NORAD {norad_id}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    break

    return results


def _parse_observation(obs: dict) -> dict:
    """Extract the fields we care about from a SatNOGS observation."""
    return {
        "observation_id": obs.get("id"),
        "norad_id": obs.get("_norad_id")
        or obs.get("satellite", {}).get("norad_cat_id"),
        "start_ts": obs.get("start", ""),
        "end_ts": obs.get("end", ""),
        "ground_station": obs.get("ground_station"),
        "vetted_status": obs.get("vetted_status", "unknown"),
        "frequency_hz": obs.get("transmitter_downlink_low"),
        "has_waterfall": bool(obs.get("waterfall")),
        "has_audio": bool(obs.get("archive_url")),
    }


async def run_satnogs_fetcher(
    store: StarlinkStore,
    interval: int = FETCH_INTERVAL,
) -> None:
    """Periodically fetch SatNOGS observations for satellites with TLE gaps.

    Strategy: on each cycle, identify satellites with TLE gaps >24h (from
    our gap detector), then query SatNOGS for recent observations of those
    specific satellites. This is targeted — we don't fetch observations for
    all 10k+ Starlink sats, only the ones that are already flagged as
    potentially in trouble.
    """
    print(f"[SATNOGS] Fetcher starting (interval={interval}s)")

    # Also import gap detection
    try:
        from services.brain.orbital_analyzer import detect_tle_gaps
    except ImportError:
        from brain.orbital_analyzer import detect_tle_gaps  # type: ignore

    cycle = 0
    while True:
        cycle += 1
        t0 = time.perf_counter()

        try:
            # Find satellites with TLE gaps
            gaps = detect_tle_gaps(store)
            if not gaps:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if cycle <= 2:
                    print(
                        f"[SATNOGS][{cycle:04d}] no TLE gaps, nothing to check ({elapsed_ms}ms)"
                    )
                await _sleep_cycle(cycle, interval)
                continue

            norad_ids = [g["norad_id"] for g in gaps[:50]]  # cap at 50 to be polite
            since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

            observations = await fetch_observations(norad_ids, since=since)
            parsed = [_parse_observation(obs) for obs in observations]

            # Store observations
            new_count = store.upsert_satnogs_observations(parsed)

            # Summarize
            from collections import Counter

            status_counts = Counter(o["vetted_status"] for o in parsed)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            print(
                f"[SATNOGS][{cycle:04d}] checked {len(norad_ids)} gapped sats, "
                f"{len(parsed)} observations ({new_count} new), "
                f"statuses: {dict(status_counts)}, {elapsed_ms}ms"
            )

            # Alert: satellites with gap AND no good observations
            for gap in gaps[:10]:
                nid = gap["norad_id"]
                sat_obs = [o for o in parsed if o["norad_id"] == nid]
                good = sum(1 for o in sat_obs if o["vetted_status"] == "good")
                failed = sum(
                    1 for o in sat_obs if o["vetted_status"] in ("failed", "bad")
                )
                if sat_obs and failed > 0 and good == 0:
                    print(
                        f"[SATNOGS][{cycle:04d}] ⚠ {gap.get('name') or nid}: "
                        f"TLE gap {gap['gap_hours']:.0f}h + {failed} failed RF observations "
                        f"— possible satellite failure"
                    )

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"[SATNOGS][{cycle:04d}] error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

        await _sleep_cycle(cycle, interval)


async def _sleep_cycle(cycle: int, interval: int) -> None:
    if cycle == 1:
        await asyncio.sleep(30)  # first cycle: short wait, let TLE fetcher go first
    else:
        await asyncio.sleep(interval)
