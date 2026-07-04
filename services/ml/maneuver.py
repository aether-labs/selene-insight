"""Maneuver detection and reconstruction using Helmholtz residual anomalies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class HelmholtzManeuverDetector:
    """Detects and isolates active maneuvers from observed orbital residuals.

    Uses a trained HelmholtzResidualModel to compute expected conservative/dissipative
    residuals, flags statistically significant deviations using Median Absolute Deviation (MAD),
    and reconstructs the isolated delta-V magnitude.
    """

    def __init__(self, helmholtz_model: nn.Module, n_mad: float = 8.0, min_threshold: float = 2.5e-5):
        """
        Args:
            helmholtz_model: Trained HelmholtzResidualModel
            n_mad: Deviation multiplier for outlier detection
            min_threshold: Minimum residual threshold to prevent noise triggering
        """
        self.model = helmholtz_model
        self.n_mad = n_mad
        self.min_threshold = min_threshold

    def detect_maneuvers(
        self,
        states: torch.Tensor,
        observed_accels: torch.Tensor,
        times: torch.Tensor,
        dt: float,
    ) -> list[dict[str, float]]:
        """Identifies anomalies in residuals against model predictions.

        Args:
            states: Tensor of shape (N, 4) or (N, 6) containing orbital states
            observed_accels: Tensor of shape (N, 2) or (N, 3) containing observed residual accelerations
            times: Tensor of shape (N,) containing time steps in seconds
            dt: Time step duration in seconds

        Returns:
            detected_maneuvers: List of isolated burns with keys:
              - 'time_s': float - time of the maneuver
              - 'delta_v_mps': float - estimated delta-V magnitude (m/s)
              - 'score': float - peak acceleration anomaly score (m/s^2)
        """
        # Run forward pass of model to predict residual acceleration
        with torch.no_grad():
            pred_accels = self.model(states)

        # Anomaly score is L2 difference between observed and predicted acceleration
        anomaly = torch.linalg.norm(observed_accels - pred_accels, dim=-1)
        anomaly_np = anomaly.cpu().numpy()
        times_np = times.cpu().numpy()

        # Calculate robust threshold via Median Absolute Deviation (MAD)
        baseline = np.median(anomaly_np)
        mad = np.median(np.abs(anomaly_np - baseline)) + 1e-12
        threshold = max(float(baseline + self.n_mad * mad), self.min_threshold)

        candidate_idx = np.where(anomaly_np > threshold)[0]
        detected: list[dict[str, float]] = []

        # Group and isolate local peaks within 1.5 * dt window
        for idx in candidate_idx:
            time_val = float(times_np[idx])
            score_val = float(anomaly_np[idx])
            dv_val = float(anomaly_np[idx] * dt)  # Delta-V = anomaly * dt

            # If we detect consecutive ticks, keep the peak score
            if detected and abs(time_val - detected[-1]["time_s"]) < 1.5 * dt:
                if score_val > detected[-1]["score"]:
                    detected[-1] = {
                        "time_s": time_val,
                        "delta_v_mps": dv_val,
                        "score": score_val,
                    }
                continue

            detected.append({
                "time_s": time_val,
                "delta_v_mps": dv_val,
                "score": score_val,
            })

        return detected
