"""Telemetry data models for Artemis II tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TelemetryPoint:
    """A single telemetry reading from the Orion spacecraft."""

    timestamp: float  # Unix epoch seconds
    met: str  # Mission Elapsed Time string (e.g. "001:05:23:41")
    phase: str  # Mission phase (e.g. "Outbound Coast")
    velocity_kms: float  # km/s
    earth_dist_km: float  # Distance from Earth in km
    moon_dist_km: float  # Distance from Moon in km

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "met": self.met,
            "phase": self.phase,
            "velocity_kms": self.velocity_kms,
            "earth_dist_km": self.earth_dist_km,
            "moon_dist_km": self.moon_dist_km,
        }

    @property
    def key(self) -> str:
        """Hash key for Lethe storage — MET is unique per reading."""
        return f"telem:{self.met}"

    @property
    def timeseries_key(self) -> str:
        """Ordered key for time-range queries."""
        return f"ts:{self.timestamp:.3f}"
