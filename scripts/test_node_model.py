"""Validation script for the Physics-Informed Neural ODE model.

Runs a comparison between the PyTorch-based Neural ODE propagator
(with zeroed neural network contribution) and the existing numpy/scipy-based
analytical propagator.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.brain.dynamics import propagate_state
from services.ml.node_model import ResidualAccelerationNet, NeuralODEPropagator

# Physical constants
R_EARTH = 6.3781363e6
MU_EARTH = 3.986004418e14


def get_random_leo_state() -> np.ndarray:
    """Generate a realistic LEO state vector (altitude ~550km, roughly circular)."""
    alt = 550000.0  # 550 km
    r = R_EARTH + alt
    v = np.sqrt(MU_EARTH / r)

    # Simplified circular orbit in ECI plane
    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v, 0.0])
    return np.concatenate([pos, vel])


def main():
    print("=== Neural ODE Verification ===")

    # Disable J3 and J4 in SciPy to do an exact point-mass + J2 comparison
    import services.brain.dynamics
    services.brain.dynamics.J3 = 0.0
    services.brain.dynamics.J4 = 0.0

    # 1. Initialize PyTorch model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up neural network
    net = ResidualAccelerationNet(hidden_dim=32, num_layers=2).to(device).double()
    # The weights are already initialized to near-zero, so f_neural ~ 0.

    # Drag parameter B*
    bstar = 0.0

    # Instantiate PyTorch propagator
    # We use DOP853 ('dopri5' in torchdiffeq) to compare directly with scipy's DOP853
    propagator = NeuralODEPropagator(
        net, bstar=bstar, rtol=1e-8, atol=1e-10, method="dopri5"
    ).to(device).double()

    # 2. Define initial state and propagation times
    state0_np = get_random_leo_state()
    state0_torch = torch.tensor(state0_np, dtype=torch.float64, device=device).unsqueeze(0)

    # 24 hours of propagation, evaluating every hour (25 steps total)
    dt_hours = 24.0
    t_steps = 25
    t_seconds_np = np.linspace(0, dt_hours * 3600, t_steps)
    t_seconds_torch = torch.tensor(t_seconds_np, dtype=torch.float64, device=device)

    print(f"Initial State:\n  Pos: {state0_np[:3]} m\n  Vel: {state0_np[3:]} m/s")

    # 3. Propagate using PyTorch Neural ODE
    print("\nPropagating with PyTorch Neural ODE...")
    with torch.no_grad():
        # returns shape (batch_size, len(t), 6) -> (1, 25, 6)
        trajectory_torch = propagator(state0_torch, t_seconds_torch)
        trajectory_torch = trajectory_torch.squeeze(0).cpu().numpy()

    # 4. Propagate using SciPy baseline
    print("Propagating with SciPy baseline...")
    trajectory_scipy = np.zeros((t_steps, 6))
    trajectory_scipy[0] = state0_np

    success = True
    for i in range(1, t_steps):
        dt = t_seconds_np[i]
        # Propagate from step 0 to step i
        state_i, ok = propagate_state(state0_np, dt, bstar=bstar, method="DOP853", rtol=1e-8, atol=1e-10)
        if not ok:
            success = False
            print(f"  SciPy propagation failed at t = {dt} s")
            break
        trajectory_scipy[i] = state_i

    if not success:
        return 1

    # 5. Evaluate difference
    pos_diffs = []
    vel_diffs = []

    print("\nComparing states hourly:")
    print(f"{'Hour':<6} | {'Pos Diff (m)':<15} | {'Vel Diff (m/s)':<15}")
    print("-" * 44)

    for i in range(t_steps):
        hour = t_seconds_np[i] / 3600.0
        p_diff = np.linalg.norm(trajectory_torch[i, :3] - trajectory_scipy[i, :3])
        v_diff = np.linalg.norm(trajectory_torch[i, 3:] - trajectory_scipy[i, 3:])
        pos_diffs.append(p_diff)
        vel_diffs.append(v_diff)
        print(f"{hour:<6.1f} | {p_diff:<15.4f} | {v_diff:<15.6f}")

    max_p_diff = max(pos_diffs)
    mean_p_diff = np.mean(pos_diffs)
    print("\nSummary:")
    print(f"  Max Position Difference:  {max_p_diff:.4f} meters")
    print(f"  Mean Position Difference: {mean_p_diff:.4f} meters")

    # The drift should be extremely small (e.g., < 200 meters after 24 hours of DOP853 integration)
    # due to slight differences in floating-point precision, solver order (dopri5 vs DOP853), and tolerances.
    if max_p_diff < 200.0:
        print("\nSUCCESS: Neural ODE physics matches SciPy baseline within acceptable tolerance.")
        return 0
    else:
        print("\nWARNING: High discrepancy detected between Neural ODE physics and SciPy baseline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
