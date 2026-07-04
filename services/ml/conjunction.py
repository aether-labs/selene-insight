"""Conjunction value fields and Proposer-Verifier screening pipeline.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import numpy as np

class ConjunctionValueMLP(nn.Module):
    """Predicts Event-Value V(s0) = min_t ||r_rel(t)|| - r_keepout directly from initial relative state s0.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Tensor of shape (B, in_dim) containing initial relative states.
                   Assumes state is unnormalized.
        """
        # Normalization scale for LEO states
        if state.shape[-1] == 4:
            pos_norm = state[:, :2] / 5000.0
            vel_norm = state[:, 2:] / 5.0
            state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
        elif state.shape[-1] == 6:
            pos_norm = state[:, :3] / 5000.0
            vel_norm = state[:, 3:] / 5.0
            state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
        else:
            state_norm = state

        return self.net(state_norm)


class ProposerVerifierScreening:
    """Implements the proposer-verifier pipeline for conjunction screening.

    Uses a fast proposer (ConjunctionValueMLP) to filter out safe trajectories,
    sending only flagged candidates near keepout boundaries to the high-fidelity verifier.
    """

    def __init__(
        self,
        proposer: nn.Module,
        verifier_propagator: nn.Module,
        r_keepout: float = 100.0,
    ):
        """
        Args:
            proposer: Trained ConjunctionValueMLP
            verifier_propagator: High-fidelity propagator module (e.g. Neural ODE)
            r_keepout: Keepout boundary radius (meters)
        """
        self.proposer = proposer
        self.verifier = verifier_propagator
        self.r_keepout = r_keepout

    def run_screening(
        self,
        s0_batch: torch.Tensor,
        t_eval: torch.Tensor,
        screening_threshold: float = 200.0,
        true_values: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Runs the screening pipeline on a batch of initial states.

        Args:
            s0_batch: Initial states of shape (B, in_dim)
            t_eval: Evaluation timesteps for the verifier
            screening_threshold: Threshold below which candidates are flagged (meters)
            true_values: Optional ground truth min distances (minus keepout) for recall calculation

        Returns:
            results: Dict of screening metrics and outputs
        """
        B = s0_batch.shape[0]
        device = s0_batch.device

        # 1. Proposer Evaluation
        t_start_prop = time.time()
        with torch.no_grad():
            pred_scaled = self.proposer(s0_batch).squeeze(-1)
            pred_values = pred_scaled * 1000.0  # Unscale from km to m
        t_prop_total = time.time() - t_start_prop

        # 2. Flag candidates that fall below the screening threshold
        flagged_mask = pred_values <= screening_threshold
        num_flagged = torch.sum(flagged_mask).item()

        # 3. Verifier Evaluation on flagged candidates only
        t_bf_start = time.time()
        with torch.no_grad():
            # Estimate brute-force time: run verifier on entire batch
            # (We run a tiny mock pass or use a constant scale to estimate if too slow,
            # but here we measure actual verifier propagation time per case)
            if B > 0:
                # Run verifier on a subset to get time-per-case
                sample_size = min(10, B)
                _ = self.verifier(s0_batch[:sample_size], t_eval)
                t_bf_sample = time.time() - t_bf_start
                t_bf_per_case = t_bf_sample / sample_size
            else:
                t_bf_per_case = 0.0

        t_verifier_total = 0.0
        verifier_trajs = None
        if num_flagged > 0:
            s0_flagged = s0_batch[flagged_mask]
            t_start_verify = time.time()
            with torch.no_grad():
                verifier_trajs = self.verifier(s0_flagged, t_eval)
            t_verifier_total = time.time() - t_start_verify

        # Total proposer-verifier pipeline time
        t_pipeline_total = t_prop_total + t_verifier_total
        t_brute_force_estimate = B * t_bf_per_case
        speedup = t_brute_force_estimate / max(t_pipeline_total, 1e-6)

        results = {
            "flagged_mask": flagged_mask,
            "num_flagged": num_flagged,
            "proposer_time": t_prop_total,
            "verifier_time": t_verifier_total,
            "pipeline_time": t_pipeline_total,
            "speedup": speedup,
            "predicted_values": pred_values,
        }

        # If ground-truth event values are supplied, compute recall and false negative rate
        if true_values is not None:
            true_conjunction_mask = true_values <= 0.0
            num_conjunctions = torch.sum(true_conjunction_mask).item()
            
            false_negatives = torch.sum(true_conjunction_mask & (~flagged_mask)).item()
            true_positives = torch.sum(true_conjunction_mask & flagged_mask).item()

            if num_conjunctions > 0:
                recall = true_positives / num_conjunctions
                fnr = false_negatives / num_conjunctions
            else:
                recall = 1.0
                fnr = 0.0

            results.update({
                "safety_recall": recall,
                "false_negative_rate": fnr,
                "num_conjunctions": num_conjunctions,
                "false_negatives": false_negatives,
            })

        return results
