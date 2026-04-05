"""Smoke tests — verify core modules import and basic functionality."""

import time


def test_lethe_put_get():
    from services.telemetry.lethe import Lethe

    store = Lethe(max_entries=100)
    store.put("k1", {"a": 1}, timestamp=1.0)
    assert store.get("k1") == {"a": 1}
    assert store.size == 1


def test_lethe_latest():
    from services.telemetry.lethe import Lethe

    store = Lethe()
    t = time.time()
    for i in range(5):
        store.put(f"k{i}", {"i": i}, timestamp=t + i)
    latest = store.latest(3)
    assert len(latest) == 3
    assert latest[0]["i"] == 4


def test_gravity_model():
    from services.brain.gravity_model import check_anomaly

    pred = check_anomaly(
        prev_velocity_kms=2.630,
        curr_velocity_kms=2.628,
        earth_dist_km=92361,
        moon_dist_km=314735,
        dt_seconds=5,
    )
    assert pred.earth_accel_ms2 > 0
    assert pred.moon_accel_ms2 > 0


def test_skeptic_agent():
    from services.brain.skeptic_agent import SkepticAgent

    agent = SkepticAgent()
    t = time.time()
    agent.analyze({"timestamp": t, "met": "001:00:00:00", "phase": "Coast",
                    "velocity_kms": 2.63, "earth_dist_km": 92000, "moon_dist_km": 315000})
    alert = agent.analyze({"timestamp": t + 5, "met": "001:00:00:05", "phase": "Coast",
                           "velocity_kms": 2.628, "earth_dist_km": 92013, "moon_dist_km": 314987})
    # Should produce an alert (or None if within threshold)
    assert agent.stats["total"] == 1


def test_cross_validator():
    from services.brain.cross_validator import CrossValidator

    v = CrossValidator(time_tolerance_sec=60)
    t = time.time()
    v.update_issinfo({"timestamp": t, "velocity_kms": 1.57, "earth_dist_km": 190000, "moon_dist_km": 245000})
    result = v.validate({"timestamp": t + 5, "velocity_kms": 1.56, "earth_dist_km": 191000, "moon_dist_km": 244000})
    assert result is not None
    assert result.grade in ("excellent", "good", "degraded", "suspect")
    assert 0 <= result.confidence <= 1


def test_telemetry_point():
    from services.telemetry.models import TelemetryPoint

    p = TelemetryPoint(timestamp=1.0, met="001:00:00:00", phase="Coast",
                       velocity_kms=2.63, earth_dist_km=92000, moon_dist_km=315000)
    d = p.to_dict()
    assert d["velocity_kms"] == 2.63
    assert p.key == "telem:001:00:00:00"


def test_api_imports():
    from services.api.main import app
    assert app.title == "Selene-Insight API"
