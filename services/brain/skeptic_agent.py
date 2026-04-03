"""Skeptic Agent — Physics verification engine for Artemis II.

Built on the Tiphys framework. Monitors telemetry, runs gravitational
models, and raises Insight_Alerts when observed data deviates from
physical predictions.

"Don't trust the numbers. Verify the physics."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

try:
    from services.brain.gravity_model import check_anomaly, GravityPrediction
except ImportError:
    from gravity_model import check_anomaly, GravityPrediction


@dataclass
class InsightAlert:
    """Structured alert when physics don't match telemetry."""

    timestamp: float
    met: str
    alert_type: str  # "orbital_maneuver" | "sensor_anomaly" | "coast_nominal"
    confidence: float  # 0.0 - 1.0
    deviation_pct: float
    details: str
    prediction: dict

    def to_json(self) -> str:
        return json.dumps({
            "type": "Insight_Alert",
            "timestamp": self.timestamp,
            "met": self.met,
            "alert_type": self.alert_type,
            "confidence": self.confidence,
            "deviation_pct": round(self.deviation_pct, 4),
            "details": self.details,
            "prediction": self.prediction,
        }, indent=2)


class SkepticAgent:
    """Tiphys-based agent that verifies telemetry against physics.

    Maintains a sliding window of telemetry readings and runs
    gravitational anomaly detection on each new data point.
    """

    def __init__(self, anomaly_threshold_pct: float = 0.5) -> None:
        self.threshold = anomaly_threshold_pct
        self._prev_point: dict | None = None
        self._alert_count = 0
        self._nominal_count = 0

    def analyze(self, point: dict) -> InsightAlert | None:
        """Analyze a telemetry point against the previous reading.

        Args:
            point: Dict with keys: timestamp, met, phase, velocity_kms,
                   earth_dist_km, moon_dist_km.

        Returns:
            InsightAlert if deviation detected, None if nominal.
        """
        if self._prev_point is None:
            self._prev_point = point
            return None

        dt = point["timestamp"] - self._prev_point["timestamp"]
        if dt <= 0:
            self._prev_point = point
            return None

        pred = check_anomaly(
            prev_velocity_kms=self._prev_point["velocity_kms"],
            curr_velocity_kms=point["velocity_kms"],
            earth_dist_km=point["earth_dist_km"],
            moon_dist_km=point["moon_dist_km"],
            dt_seconds=dt,
            threshold_pct=self.threshold,
        )

        self._prev_point = point

        if not pred.is_anomalous:
            self._nominal_count += 1
            return None

        self._alert_count += 1
        alert_type, confidence, details = self._classify_anomaly(pred, point)

        return InsightAlert(
            timestamp=point["timestamp"],
            met=point["met"],
            alert_type=alert_type,
            confidence=confidence,
            deviation_pct=pred.deviation_pct,
            details=details,
            prediction={
                "predicted_dv_kms": round(pred.predicted_dv_kms, 6),
                "observed_dv_kms": round(pred.observed_dv_kms, 6),
                "earth_accel_ms2": round(pred.earth_accel_ms2, 6),
                "moon_accel_ms2": round(pred.moon_accel_ms2, 6),
                "net_accel_ms2": round(pred.net_accel_ms2, 6),
            },
        )

    def _classify_anomaly(
        self, pred: GravityPrediction, point: dict
    ) -> tuple[str, float, str]:
        """Classify an anomaly as maneuver or sensor issue.

        Heuristics:
        - Large, sustained dv → likely orbital maneuver (burn)
        - Sudden spike with immediate return → likely sensor glitch
        - Deviation during known burn phases → expected maneuver
        """
        phase = point.get("phase", "").lower()
        deviation = pred.deviation_pct

        # Known burn phases
        if any(kw in phase for kw in ("burn", "insertion", "correction", "tli")):
            return (
                "orbital_maneuver",
                0.95,
                f"Velocity deviation of {deviation:.2f}% during '{point['phase']}' phase — "
                f"consistent with planned orbital maneuver.",
            )

        # Large deviation during coast → likely unplanned burn or correction
        if deviation > 5.0:
            return (
                "orbital_maneuver",
                0.8,
                f"Significant velocity deviation ({deviation:.2f}%) during coast phase. "
                f"Possible mid-course correction or unplanned burn.",
            )

        # Small deviation → sensor noise or minor attitude adjustment
        return (
            "sensor_anomaly",
            0.6,
            f"Minor velocity deviation ({deviation:.2f}%) not explained by gravity model. "
            f"Possible sensor noise or RCS thruster activity.",
        )

    @property
    def stats(self) -> dict:
        return {
            "alerts": self._alert_count,
            "nominal": self._nominal_count,
            "total": self._alert_count + self._nominal_count,
        }
