"""Cross-Source Validator — compares issinfo against JPL Horizons ground truth.

"Don't trust the numbers. Verify the physics. Verify the source."

When Horizons delivers a new data point, finds the nearest issinfo reading
and computes deviation metrics. Produces a confidence score and optionally
raises data quality alerts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of cross-validating two telemetry sources."""

    timestamp: float
    issinfo_point: dict
    horizons_point: dict

    # Deviations
    velocity_pct: float      # % difference in velocity
    earth_dist_pct: float    # % difference in earth distance
    moon_dist_pct: float     # % difference in moon distance
    position_km: float       # absolute position difference (km) if 3D available

    # Overall
    confidence: float        # 0.0 - 1.0, how much we trust issinfo
    grade: str               # "excellent" | "good" | "degraded" | "suspect"
    details: str

    def to_dict(self) -> dict:
        return {
            "type": "cross_validation",
            "timestamp": self.timestamp,
            "deviations": {
                "velocity_pct": round(self.velocity_pct, 3),
                "earth_dist_pct": round(self.earth_dist_pct, 3),
                "moon_dist_pct": round(self.moon_dist_pct, 3),
                "position_km": round(self.position_km, 1),
            },
            "confidence": round(self.confidence, 4),
            "grade": self.grade,
            "details": self.details,
        }


def _pct_diff(a: float, b: float) -> float:
    """Percentage difference between two values."""
    if abs(b) < 1e-9:
        return 0.0 if abs(a) < 1e-9 else 100.0
    return abs(a - b) / abs(b) * 100


def _position_diff_km(issinfo: dict, horizons: dict) -> float:
    """3D position difference if Horizons provides pos_km vector."""
    pos = horizons.get("pos_km")
    if not pos:
        # Fall back to scalar distance comparison
        de = abs(issinfo.get("earth_dist_km", 0) - horizons.get("earth_dist_km", 0))
        dm = abs(issinfo.get("moon_dist_km", 0) - horizons.get("moon_dist_km", 0))
        return math.sqrt(de * de + dm * dm)

    # Horizons has 3D ECI position; issinfo only has distances.
    # Best we can do: compare earth_dist (radial distance from Earth).
    r_horizons = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    r_issinfo = issinfo.get("earth_dist_km", 0)
    return abs(r_horizons - r_issinfo)


class CrossValidator:
    """Compares issinfo telemetry against JPL Horizons ground truth.

    Usage:
        validator = CrossValidator()

        # Called every 5s by issinfo worker:
        validator.update_issinfo(point_dict)

        # Called every 60s by Horizons worker:
        result = validator.validate(horizons_point_dict)
    """

    def __init__(
        self,
        excellent_threshold: float = 0.5,   # < 0.5% = excellent
        good_threshold: float = 2.0,        # < 2% = good
        degraded_threshold: float = 5.0,    # < 5% = degraded, else suspect
        time_tolerance_sec: float = 10.0,   # max age diff for pairing
    ) -> None:
        self._thresholds = (excellent_threshold, good_threshold, degraded_threshold)
        self._time_tolerance = time_tolerance_sec

        # Ring buffer of recent issinfo points (keep last 30, ~2.5 min at 5s)
        self._issinfo_buffer: list[dict] = []
        self._max_buffer = 30

        # History of validation results
        self._results: list[ValidationResult] = []
        self._max_results = 100

        # Running stats
        self.total_validations = 0
        self.grades: dict[str, int] = {
            "excellent": 0, "good": 0, "degraded": 0, "suspect": 0,
        }

    def update_issinfo(self, point: dict) -> None:
        """Buffer an issinfo telemetry point."""
        self._issinfo_buffer.append(point)
        if len(self._issinfo_buffer) > self._max_buffer:
            self._issinfo_buffer = self._issinfo_buffer[-self._max_buffer:]

    def validate(self, horizons_point: dict) -> ValidationResult | None:
        """Cross-validate against the nearest issinfo reading.

        Args:
            horizons_point: Dict from Horizons worker with source="jpl_horizons".

        Returns:
            ValidationResult, or None if no issinfo data available.
        """
        if not self._issinfo_buffer:
            return None

        h_ts = horizons_point.get("timestamp", 0)

        # Find closest issinfo point by timestamp
        best = None
        best_dt = float("inf")
        for p in self._issinfo_buffer:
            dt = abs(p.get("timestamp", 0) - h_ts)
            if dt < best_dt:
                best_dt = dt
                best = p

        if best is None or best_dt > self._time_tolerance:
            return None

        # Compute deviations
        vel_pct = _pct_diff(
            best.get("velocity_kms", 0),
            horizons_point.get("velocity_kms", 0),
        )
        earth_pct = _pct_diff(
            best.get("earth_dist_km", 0),
            horizons_point.get("earth_dist_km", 0),
        )
        moon_pct = _pct_diff(
            best.get("moon_dist_km", 0),
            horizons_point.get("moon_dist_km", 0),
        )
        pos_km = _position_diff_km(best, horizons_point)

        # Max deviation drives the grade
        max_dev = max(vel_pct, earth_pct, moon_pct)
        exc, good, deg = self._thresholds

        if max_dev < exc:
            grade = "excellent"
            confidence = 1.0
            details = f"All metrics within {exc}%. Sources highly consistent."
        elif max_dev < good:
            grade = "good"
            confidence = 0.95 - (max_dev - exc) / (good - exc) * 0.1
            details = f"Max deviation {max_dev:.2f}%. Sources consistent."
        elif max_dev < deg:
            grade = "degraded"
            confidence = 0.8 - (max_dev - good) / (deg - good) * 0.3
            details = (
                f"Max deviation {max_dev:.2f}%. Possible scraping drift or "
                f"timing mismatch. Horizons data preferred."
            )
        else:
            grade = "suspect"
            confidence = max(0.1, 0.5 - (max_dev - deg) / 50)
            details = (
                f"Max deviation {max_dev:.2f}%. issinfo data unreliable. "
                f"Falling back to Horizons as primary source."
            )

        result = ValidationResult(
            timestamp=time.time(),
            issinfo_point=best,
            horizons_point=horizons_point,
            velocity_pct=vel_pct,
            earth_dist_pct=earth_pct,
            moon_dist_pct=moon_pct,
            position_km=pos_km,
            confidence=confidence,
            grade=grade,
            details=details,
        )

        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results:]

        self.total_validations += 1
        self.grades[grade] += 1

        return result

    @property
    def latest_result(self) -> ValidationResult | None:
        return self._results[-1] if self._results else None

    @property
    def stats(self) -> dict:
        latest = self.latest_result
        return {
            "total_validations": self.total_validations,
            "grades": dict(self.grades),
            "latest_confidence": round(latest.confidence, 4) if latest else None,
            "latest_grade": latest.grade if latest else None,
        }

    @property
    def recent_results(self) -> list[dict]:
        return [r.to_dict() for r in self._results[-20:]]
