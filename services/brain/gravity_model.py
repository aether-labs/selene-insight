"""Gravitational model for Artemis II trajectory verification.

Computes theoretical velocity changes based on Earth and Moon gravity.
Used by the Skeptic Agent to detect orbital maneuvers or sensor anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Physical constants
G = 6.674e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Earth mass (kg)
M_MOON = 7.342e22  # Moon mass (kg)
R_EARTH = 6.371e6  # Earth radius (m)
R_MOON = 1.737e6  # Moon radius (m)


@dataclass
class GravityPrediction:
    """Predicted gravitational effects on the spacecraft."""

    earth_accel_ms2: float  # Acceleration due to Earth gravity (m/s^2)
    moon_accel_ms2: float  # Acceleration due to Moon gravity (m/s^2)
    net_accel_ms2: float  # Net radial acceleration (m/s^2)
    predicted_dv_kms: float  # Predicted delta-v over interval (km/s)
    observed_dv_kms: float  # Observed delta-v from telemetry (km/s)
    deviation_pct: float  # Percentage deviation
    is_anomalous: bool  # True if deviation > threshold


def gravitational_acceleration(mass: float, distance_km: float) -> float:
    """Calculate gravitational acceleration at a given distance.

    Args:
        mass: Mass of the body (kg).
        distance_km: Distance from the body's center (km).

    Returns:
        Acceleration in m/s^2.
    """
    r = distance_km * 1000  # Convert to meters
    if r < 1:
        return 0.0
    return G * mass / (r * r)


def predict_velocity_change(
    earth_dist_km: float,
    moon_dist_km: float,
    dt_seconds: float,
    velocity_kms: float,
) -> GravityPrediction:
    """Predict the velocity change due to gravity over a time interval.

    This is a simplified 1D model — assumes the spacecraft is on the
    Earth-Moon line. Good enough for anomaly detection (not navigation).

    Args:
        earth_dist_km: Distance from Earth center (km).
        moon_dist_km: Distance from Moon center (km).
        dt_seconds: Time interval (seconds).
        velocity_kms: Current velocity (km/s).

    Returns:
        GravityPrediction with theoretical and observed values.
    """
    a_earth = gravitational_acceleration(M_EARTH, earth_dist_km)
    a_moon = gravitational_acceleration(M_MOON, moon_dist_km)

    # On outbound coast: Earth decelerates, Moon accelerates
    # Sign convention: positive = away from Earth
    net_accel = a_moon - a_earth  # simplified radial model

    # Predicted delta-v (km/s)
    predicted_dv = (net_accel * dt_seconds) / 1000

    return GravityPrediction(
        earth_accel_ms2=a_earth,
        moon_accel_ms2=a_moon,
        net_accel_ms2=net_accel,
        predicted_dv_kms=predicted_dv,
        observed_dv_kms=0.0,  # filled by caller
        deviation_pct=0.0,
        is_anomalous=False,
    )


def check_anomaly(
    prev_velocity_kms: float,
    curr_velocity_kms: float,
    earth_dist_km: float,
    moon_dist_km: float,
    dt_seconds: float,
    threshold_pct: float = 0.5,
) -> GravityPrediction:
    """Compare observed velocity change against gravitational prediction.

    Args:
        prev_velocity_kms: Previous velocity reading (km/s).
        curr_velocity_kms: Current velocity reading (km/s).
        earth_dist_km: Current Earth distance (km).
        moon_dist_km: Current Moon distance (km).
        dt_seconds: Time between readings (seconds).
        threshold_pct: Deviation threshold to flag anomaly (default 0.5%).

    Returns:
        GravityPrediction with anomaly flag.
    """
    pred = predict_velocity_change(
        earth_dist_km, moon_dist_km, dt_seconds, curr_velocity_kms
    )

    observed_dv = curr_velocity_kms - prev_velocity_kms
    pred.observed_dv_kms = observed_dv

    if abs(pred.predicted_dv_kms) > 1e-9:
        pred.deviation_pct = (
            abs(observed_dv - pred.predicted_dv_kms) / abs(pred.predicted_dv_kms) * 100
        )
    elif abs(observed_dv) > 1e-6:
        pred.deviation_pct = 100.0  # predicted zero change but observed non-zero
    else:
        pred.deviation_pct = 0.0

    pred.is_anomalous = pred.deviation_pct > threshold_pct

    return pred
