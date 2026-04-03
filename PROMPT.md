# Project Selene-Insight — Continuation Prompt

## What is this project?

Artemis II real-time digital twin and deep analysis system. An AI agent monitors spacecraft telemetry, verifies it against physics models, and visualizes the trajectory in 3D.

Core philosophy: "Don't trust the numbers. Verify the physics."

## What's already built

### 1. Telemetry Worker (`services/telemetry/telemetry_worker.py`)
- **Working.** Playwright headless scrapes https://issinfo.net/artemis.html every 5 seconds.
- DOM selectors: `.sp-section` containers with `.sp-label` + `.sp-val` pairs, `.sp-met-value` for MET.
- Extracts: Velocity (km/s), Earth Distance (km), Moon Distance (km), MET, Phase.
- Regex fallback if structured extraction misses values.
- Tested live — successfully pulled real data: `v=2.630 km/s, Earth=92361 km, Moon=314735 km, Phase=Outbound Coast`.

### 2. Lethe KV Store (`services/telemetry/lethe.py`)
- In-memory hash-indexed KV store. Point lookup benchmarked at **2.0 us** (target was < 1ms).
- Two access patterns: O(1) point lookup by key, O(log n + k) time-range scan via ordered timestamp index.
- Eviction policy: oldest 10% when over max_entries (default 500k).
- Thread-safe with lock.

### 3. Gravity Model (`services/brain/gravity_model.py`)
- Earth/Moon gravitational acceleration calculator.
- `check_anomaly()` compares observed delta-v against theoretical prediction.
- Simplified 1D model (Earth-Moon line). Sufficient for anomaly detection, not navigation.
- Tested: correctly flags large velocity deviations.

### 4. Skeptic Agent (`services/brain/skeptic_agent.py`)
- Analyzes consecutive telemetry points.
- Classifies anomalies: `orbital_maneuver` vs `sensor_anomaly` based on deviation magnitude and mission phase.
- Produces structured `InsightAlert` JSON.
- Tested with synthetic data — correctly identifies burns and sensor noise.

### 5. Data Model (`services/telemetry/models.py`)
- `TelemetryPoint` dataclass with `to_dict()`, `key` (for Lethe point lookup), `timeseries_key` (for range scan).

## Project structure

```
selene-insight/
├── services/
│   ├── telemetry/
│   │   ├── telemetry_worker.py   # Playwright scraper + main loop
│   │   ├── lethe.py              # Hash-indexed KV store
│   │   └── models.py             # TelemetryPoint dataclass
│   └── brain/
│       ├── skeptic_agent.py      # Physics verification agent
│       └── gravity_model.py      # Gravitational model
├── apps/
│   └── web/
│       └── src/                  # (empty — CesiumJS dashboard not started)
└── PROMPT.md                     # This file
```

## What's NOT built yet

### CesiumJS 3D Dashboard (`apps/web/`)
- React + CesiumJS component showing Earth-Moon system at 1:1 scale.
- CZML path for Orion trajectory (dynamic, fed from Lethe).
- TimeSlider component for "rewind" — queries Lethe's time-range index.
- WebSocket connection to receive live telemetry updates.

### Skeptic Agent live integration
- The Skeptic Agent works in isolation but is not yet running in the telemetry_worker's main loop when using Playwright. The `_run_skeptic()` hook exists in telemetry_worker.py but hasn't been tested with live data end-to-end.
- To test: run `telemetry_worker.py` with `PYTHONPATH` including both `services/telemetry` and `services/brain`.

### API layer
- No REST API yet. Lethe is in-process only. Need a FastAPI service exposing:
  - `GET /api/telemetry/latest` — latest N readings
  - `GET /api/telemetry/range?start=&end=` — time range query
  - `GET /api/alerts/latest` — recent Skeptic Agent alerts
  - `WebSocket /ws/telemetry` — live stream

### GitHub Actions / K8s
- No CI/CD or deployment configs yet.
- Target: monorepo with `/services/telemetry`, `/services/brain`, `/apps/web` each as a Docker container.

## How to run what exists

```bash
# From the tiphys project (has playwright installed):
cd ~/projects/tiphys

# Test live telemetry scraping (3 cycles):
PYTHONPATH=~/projects/selene-insight/services/telemetry uv run python -c "
import asyncio
from telemetry_worker import scrape_telemetry, store, ARTEMIS_URL
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(ARTEMIS_URL, wait_until='domcontentloaded', timeout=30000)
        await asyncio.sleep(5)
        for i in range(3):
            point = await scrape_telemetry(page)
            if point:
                print(f'MET={point.met} v={point.velocity_kms:.3f} km/s Earth={point.earth_dist_km:.0f} Moon={point.moon_dist_km:.0f}')
            await asyncio.sleep(5)
        await browser.close()

asyncio.run(test())
"

# Test gravity model + Skeptic Agent (no network needed):
PYTHONPATH=~/projects/selene-insight/services/telemetry:~/projects/selene-insight/services/brain python3 -c "
from skeptic_agent import SkepticAgent
import time
agent = SkepticAgent()
t = time.time()
p1 = {'timestamp': t, 'met': '001:05:23:41', 'phase': 'Outbound Coast', 'velocity_kms': 2.630, 'earth_dist_km': 92361, 'moon_dist_km': 314735}
p2 = {'timestamp': t+5, 'met': '001:05:23:46', 'phase': 'Outbound Coast', 'velocity_kms': 2.628, 'earth_dist_km': 92374, 'moon_dist_km': 314722}
agent.analyze(p1)
alert = agent.analyze(p2)
print(f'Alert: {alert}')
"
```

## Suggested next steps (in priority order)

1. **Skeptic Agent live test** — Run telemetry_worker.py with Skeptic Agent connected, observe alerts on real data.
2. **FastAPI service** — Expose Lethe data via REST + WebSocket for the dashboard.
3. **CesiumJS dashboard** — React app with Earth-Moon-Orion visualization and rewind slider.
4. **Monorepo packaging** — pyproject.toml, Docker, GitHub Actions.

## Technical notes

- Playwright requires `playwright install chromium` after pip install.
- The issinfo.net page uses JavaScript to populate `.sp-val` elements — must wait for JS hydration before scraping.
- Lethe is in-memory only. For persistence across restarts, consider writing snapshots to disk or switching to Redis.
- The gravity model is 1D (radial only). For higher fidelity, use JPL ephemeris data (via `astropy` or `jplephem`).
