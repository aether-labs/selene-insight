"""Smoke tests for Starlink constellation tracker."""

import tempfile
import time
import os


def _make_store():
    from services.telemetry.store import StarlinkStore
    db = tempfile.mktemp(suffix=".db")
    return StarlinkStore(db), db


SAMPLE_TLES = [
    {
        "norad_id": 44714, "epoch_jd": 2460400.5,
        "line1": "1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023",
        "line2": "2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14",
        "name": "STARLINK-1008", "inclination": 53.15, "mean_motion": 15.347,
        "eccentricity": 0.000348, "shell_km": 550, "intl_designator": "19074B",
        "launch_group": "19074",
    },
    {
        "norad_id": 44718, "epoch_jd": 2460400.5,
        "line1": "1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020",
        "line2": "2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15",
        "name": "STARLINK-1012", "inclination": 53.16, "mean_motion": 15.340,
        "eccentricity": 0.000391, "shell_km": 550, "intl_designator": "19074F",
        "launch_group": "19074",
    },
]


def test_store_crud():
    store, db = _make_store()
    n = store.upsert_tles(SAMPLE_TLES)
    assert n == 2
    assert store.stats["satellites"] == 2
    assert store.stats["tle_records"] == 2

    # Dedup: re-inserting the same TLEs must return 0 new rows (regression
    # guard for the old conn.total_changes bug that counted every iteration).
    n2 = store.upsert_tles(SAMPLE_TLES)
    assert n2 == 0
    assert store.stats["tle_records"] == 2

    latest = store.get_latest_tles()
    assert len(latest) == 2

    sat = store.get_satellite(44714)
    assert sat["name"] == "STARLINK-1008"

    os.unlink(db)


def test_store_anomaly():
    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "altitude_before_km": 550, "altitude_after_km": 540,
        "cause": "maneuver", "confidence": 0.8, "classified_by": "delta_v_rule",
    })
    anomalies = store.get_anomalies()
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly_type"] == "altitude_change"
    assert anomalies[0]["cause"] == "maneuver"
    assert anomalies[0]["confidence"] == 0.8
    assert anomalies[0]["classified_by"] == "delta_v_rule"
    os.unlink(db)


def test_store_fetch_log():
    store, db = _make_store()
    store.log_fetch(
        status="ok", http_bytes=12345, parsed_count=100,
        new_tle_count=42, parse_errors=0, duration_ms=830,
        raw_archive_path="data/raw/20260412/celestrak-20260412-120000.tle.gz",
    )
    store.log_fetch(
        status="error", duration_ms=5000,
        error_msg="ReadTimeout: timed out",
    )
    log = store.get_fetch_log()
    assert len(log) == 2
    # Newest first
    assert log[0]["status"] == "error"
    assert log[0]["error_msg"].startswith("ReadTimeout")
    assert log[1]["status"] == "ok"
    assert log[1]["new_tle_count"] == 42
    assert store.stats["fetches"] == 2
    os.unlink(db)


def test_store_migration_idempotent():
    """Running _init_db twice on the same file must not error and must preserve data."""
    from services.telemetry.store import StarlinkStore, SCHEMA_VERSION
    import sqlite3

    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)
    store.log_fetch(status="ok", parsed_count=2, new_tle_count=2)

    # Re-open the same DB — _migrate() must be a no-op on already-current schema.
    store2 = StarlinkStore(db)
    assert store2.stats["tle_records"] == 2
    assert store2.stats["fetches"] == 1

    # PRAGMA user_version should equal SCHEMA_VERSION.
    conn = sqlite3.connect(db)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == SCHEMA_VERSION

    os.unlink(db)


def test_store_migration_from_v1():
    """A v1 DB (no user_version, no fetch_log, no cause column) must migrate forward."""
    from services.telemetry.store import StarlinkStore, SCHEMA_VERSION
    import sqlite3

    db = tempfile.mktemp(suffix=".db")
    # Hand-build a v1 schema without fetch_log / cause.
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norad_id INTEGER NOT NULL,
            epoch_jd REAL NOT NULL,
            fetched_at REAL NOT NULL,
            line1 TEXT NOT NULL,
            line2 TEXT NOT NULL,
            inclination REAL,
            mean_motion REAL,
            eccentricity REAL,
            UNIQUE(norad_id, epoch_jd)
        );
        CREATE TABLE satellite (
            norad_id INTEGER PRIMARY KEY,
            name TEXT,
            intl_designator TEXT,
            shell_km REAL,
            launch_group TEXT,
            first_seen REAL,
            last_seen REAL,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE anomaly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norad_id INTEGER NOT NULL,
            detected_at REAL NOT NULL,
            anomaly_type TEXT NOT NULL,
            details TEXT,
            altitude_before_km REAL,
            altitude_after_km REAL
        );
    """)
    conn.execute(
        "INSERT INTO tle (norad_id, epoch_jd, fetched_at, line1, line2) VALUES (?, ?, ?, ?, ?)",
        (44714, 2460400.5, time.time(), "line1", "line2"),
    )
    conn.commit()
    conn.close()

    # Opening via StarlinkStore should migrate forward.
    store = StarlinkStore(db)
    assert store.stats["tle_records"] == 1
    assert store.stats["fetches"] == 0

    # fetch_log must exist and be writable.
    store.log_fetch(status="ok", parsed_count=1, new_tle_count=0)
    assert store.stats["fetches"] == 1

    # anomaly.cause column must exist.
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "decay", "cause": "natural_decay",
    })
    anoms = store.get_anomalies()
    assert anoms[0]["cause"] == "natural_decay"

    conn = sqlite3.connect(db)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == SCHEMA_VERSION

    os.unlink(db)


def test_propagator():
    from services.telemetry.propagator import Propagator

    prop = Propagator()
    loaded = prop.load_tles(SAMPLE_TLES)
    assert loaded == 2

    positions = prop.propagate_all()
    assert len(positions) == 2
    for p in positions:
        assert -90 <= p["lat"] <= 90
        assert -180 <= p["lon"] <= 180
        assert p["alt_km"] > 100


def test_propagator_performance():
    """Ensure batch propagation is fast enough."""
    from services.telemetry.propagator import Propagator

    prop = Propagator()
    # Create 100 copies with different IDs
    tles = []
    for i in range(100):
        t = dict(SAMPLE_TLES[0])
        t["norad_id"] = 90000 + i
        tles.append(t)
    prop.load_tles(tles)

    t0 = time.perf_counter()
    positions = prop.propagate_all()
    elapsed = time.perf_counter() - t0

    assert len(positions) == 100
    assert elapsed < 1.0  # should be << 1s


def test_tle_parser():
    from services.telemetry.tle_fetcher import parse_tle_text

    text = """STARLINK-1008
1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023
2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14
STARLINK-1012
1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020
2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15"""

    tles, errors = parse_tle_text(text)
    assert len(tles) == 2
    assert errors == 0
    assert tles[0]["norad_id"] == 44714
    assert tles[0]["name"] == "STARLINK-1008"
    assert tles[0]["inclination"] > 50


def test_tle_parser_counts_errors():
    """Malformed triplets must be counted, not silently dropped."""
    from services.telemetry.tle_fetcher import parse_tle_text

    text = """STARLINK-1008
1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023
2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14
GARBAGE-SAT
NOT-A-TLE-LINE
ALSO-NOT-A-TLE"""

    tles, errors = parse_tle_text(text)
    assert len(tles) == 1
    assert errors >= 1


def test_archive_raw(tmp_path):
    from services.telemetry.tle_fetcher import archive_raw
    import gzip

    path = archive_raw("STARLINK-1008\nline1\nline2\n", raw_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".gz"
    # Round-trip: file must be gzipped and contain the original text.
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert "STARLINK-1008" in f.read()


def test_orbital_analyzer():
    from services.brain.orbital_analyzer import analyze_tle_pair

    old = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003, "epoch_jd": 2460400.0}
    new = {"norad_id": 44714, "mean_motion": 15.50, "eccentricity": 0.0003, "epoch_jd": 2460401.0}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] in ("altitude_change", "deorbiting", "reentry")


def test_api_imports():
    from services.api.main import app
    assert app.title == "ArgusOrb API"
