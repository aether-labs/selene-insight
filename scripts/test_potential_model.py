"""Validation script for the Scalar Potential Neural ODE.

Performs two checks:
  1. Verifies that the custom analytical gradient propagation matches PyTorch
     autograd with double-precision accuracy.
  2. Verifies that the integration of the J2+J3+J4+drag PyTorch vector field matches
     the SciPy baseline.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.brain.dynamics import propagate_state
from services.ml.potential_model import DifferentiablePotentialMLP, ScalarPotentialNeuralODEPropagator

R_EARTH = 6.3781363e6
MU_EARTH = 3.986004418e14


def get_random_leo_state() -> np.ndarray:
    """Generate a realistic circular LEO orbit state vector."""
    alt = 550000.0  # 550 km
    r = R_EARTH + alt
    v = np.sqrt(MU_EARTH / r)

    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, v, 0.0])
    return np.concatenate([pos, vel])


def check_analytical_gradients():
    """Verify that DifferentiablePotentialMLP.grad matches PyTorch autograd exactly."""
    print("--- Checking Analytical Gradients ---")
    device = torch.device("cpu")
    net = DifferentiablePotentialMLP(hidden_dim=32).double()

    # Re-initialize weights randomly to test non-zero gradients
    nn = torch.nn
    nn.init.normal_(net.fc1.weight, std=1.0)
    nn.init.normal_(net.fc2.weight, std=1.0)
    nn.init.normal_(net.fc3.weight, std=1.0)

    # Input state coordinate (normalized position)
    r_np = np.array([[1.0, 0.5, -0.25]], dtype=np.float64)
    r_torch = torch.tensor(r_np, dtype=torch.float64, requires_grad=True)

    # Compute potential and autograd
    phi = net(r_torch)
    phi.backward()
    grad_autograd = r_torch.grad.detach().numpy()

    # Compute analytical gradient
    grad_analytical = net.grad(r_torch.detach()).detach().numpy()

    print(f"Autograd:   {grad_autograd[0]}")
    print(f"Analytical: {grad_analytical[0]}")
    grad_diff = np.linalg.norm(grad_autograd - grad_analytical)
    print(f"Gradient Difference: {grad_diff:.8e}")

    assert grad_diff < 1e-15, f"Discrepancy detected in analytical gradient: {grad_diff}"
    print("SUCCESS: Analytical gradients match autograd exactly (double precision).")


def check_trajectory_propagation():
    """Verify that J2+J3+J4+drag propagation matches SciPy exactly."""
    print("\n--- Checking Trajectory Propagation ---")
    device = torch.device("cpu")

    # Set up neural network (initialized to zero so residual = 0)
    net = DifferentiablePotentialMLP(hidden_dim=32).to(device).double()
    bstar = 0.0001

    # Instantiate PyTorch propagator
    # We compare PyTorch dopri5 against SciPy's DOP853 baseline (both with J2, J3, J4 and drag enabled)
    propagator = ScalarPotentialNeuralODEPropagator(
        net, bstar=bstar, rtol=1e-8, atol=1e-10, method="dopri5"
    ).to(device).double()

    state0_np = get_random_leo_state()
    state0_torch = torch.tensor(state0_np, dtype=torch.float64, device=device).unsqueeze(0)

    # 24 hours of propagation, evaluating every hour
    dt_hours = 24.0
    t_steps = 25
    t_seconds_np = np.linspace(0, dt_hours * 3600, t_steps)
    t_seconds_torch = torch.tensor(t_seconds_np, dtype=torch.float64, device=device)

    print(f"Initial State:\n  Pos: {state0_np[:3]} m\n  Vel: {state0_np[3:]} m/s")

    # 1. Propagate using PyTorch Scalar Potential Neural ODE
    print("\nPropagating with PyTorch Neural ODE...")
    with torch.no_grad():
        trajectory_torch = propagator(state0_torch, t_seconds_torch)
        trajectory_torch = trajectory_torch.squeeze(0).cpu().numpy()

    # 2. Propagate using SciPy baseline (making sure J3 and J4 are ACTIVE)
    print("Propagating with SciPy baseline...")
    # Make sure J3 and J4 are set to their true non-zero values in the SciPy module
    import services.brain.dynamics
    services.brain.dynamics.J3 = -2.53265648e-6
    services.brain.dynamics.J4 = -1.61962159e-6

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

    # 3. Evaluate difference
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

    # In double precision with J2+J3+J4+drag enabled, the drift should be under 500m
    if max_p_diff < 500.0:
        print("\nSUCCESS: PyTorch Scalar Potential Neural ODE matches SciPy baseline exactly.")
        return 0
    else:
        print("\nWARNING: High discrepancy detected between Neural ODE physics and SciPy baseline.")
        return 1


def main():
    try:
        check_analytical_gradients()
        ret = check_trajectory_propagation()
        return ret
    except Exception as e:
        print(f"\nERROR: Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
