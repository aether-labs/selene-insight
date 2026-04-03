"""Combined runner — starts FastAPI server + telemetry workers in one process.

Usage:
    python -m services.api.runner
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import uvicorn

from services.api.main import (
    app, store, alert_store, validator,
    ingest_telemetry, ingest_alert,
    broadcast_telemetry, broadcast_alert,
)


async def run_all() -> None:
    """Run API server, issinfo scraper, Horizons worker, cross-validator."""
    # Start uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Start telemetry worker (issinfo.net scraper)
    from services.telemetry.telemetry_worker import run_worker
    import services.telemetry.telemetry_worker as tw
    tw.store = store

    # Start Horizons worker (JPL ephemeris)
    from services.telemetry.horizons_worker import run_horizons_worker

    def on_issinfo_telemetry(point: dict) -> None:
        """Feed issinfo data into the cross-validator buffer."""
        validator.update_issinfo(point)

    def on_horizons_telemetry(point: dict) -> None:
        """Ingest Horizons data and run cross-validation."""
        ingest_telemetry(point)

        # Cross-validate against latest issinfo
        result = validator.validate(point)
        if result:
            grade = result.grade
            conf = result.confidence
            details = result.details
            emoji = {"excellent": "+", "good": "~", "degraded": "!", "suspect": "X"}
            print(
                f"  [{emoji.get(grade, '?')}VALIDATE] {grade.upper()} "
                f"(confidence={conf:.1%}) "
                f"vel={result.velocity_pct:.2f}% "
                f"earth={result.earth_dist_pct:.2f}% "
                f"moon={result.moon_dist_pct:.2f}%"
            )

            # Broadcast validation result to dashboard
            asyncio.create_task(
                _broadcast_validation(result.to_dict())
            )

            # If suspect, raise alert
            if grade == "suspect":
                alert = {
                    "type": "Insight_Alert",
                    "timestamp": result.timestamp,
                    "met": point.get("met", ""),
                    "alert_type": "data_quality",
                    "confidence": conf,
                    "deviation_pct": max(
                        result.velocity_pct,
                        result.earth_dist_pct,
                        result.moon_dist_pct,
                    ),
                    "details": details,
                }
                ingest_alert(alert)
                asyncio.create_task(broadcast_alert(alert))

    # Hook into the issinfo worker's data flow
    original_ingest = ingest_telemetry

    def hooked_ingest(point: dict) -> None:
        original_ingest(point)
        if point.get("source", "issinfo") == "issinfo":
            on_issinfo_telemetry(point)

    # Monkey-patch ingest to capture issinfo data for validator
    import services.api.main as api_main
    api_main.ingest_telemetry = hooked_ingest

    tasks = [
        asyncio.create_task(server.serve()),
        asyncio.create_task(run_worker(with_skeptic=True, api_mode=True)),
        asyncio.create_task(run_horizons_worker(
            on_telemetry=on_horizons_telemetry,
            poll_interval=60,
        )),
    ]

    await asyncio.gather(*tasks)


async def _broadcast_validation(result: dict) -> None:
    """Push cross-validation result to WebSocket clients."""
    from services.api.main import _ws_clients
    if not _ws_clients:
        return
    import json
    message = json.dumps({"type": "validation", "data": result})
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def main() -> None:
    def _shutdown(sig: int, frame: object) -> None:
        print(f"\n[STOP] Signal {sig}. Entries: {store.size}, Alerts: {alert_store.size}")
        print(f"[STOP] Validation stats: {validator.stats}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
