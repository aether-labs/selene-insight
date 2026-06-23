"""Benchmark script comparing PI-NODE, Scalar Potential, and HNN models.

This script:
1. Generates a synthetic LEO trajectory under J2 gravity, drag, and a small
   constant out-of-plane residual acceleration.
2. Trains four models:
   - PI-NODE (Standard non-conservative MLP)
   - Scalar Potential (Conservative neural force field)
   - Separable HNN (Equivalent to Scalar Potential, momentum-separated)
   - Non-Separable HNN (Fully coupled latent space Hamiltonian)
3. Evaluates them on:
   - Training loss convergence
   - Long-term orbital energy conservation (Hamiltonian drift)
   - Integration drift (position MSE against ground truth over 24 hours)
   - Time-reversibility error
   - Inference speed (flops/forward pass time)
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

from services.ml.node_model import ResidualAccelerationNet, NeuralODEVectorField
from services.ml.potential_model import DifferentiablePotentialMLP, ScalarPotentialNeuralODEVectorField
from services.ml.hamiltonian_model import DifferentiableHamiltonianMLP, HamiltonianNeuralODEVectorField

# Physical Constants
MU_EARTH = 3.986004418e14
R_EARTH = 6.3781363e6
V_NORM = 7500.0


def generate_benchmark_trajectory(n_steps: int = 500, dt_s: float = 60.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a high-fidelity synthetic trajectory under perturbed physics.
    
    Includes J2 gravity, drag, and a constant out-of-plane thrust (non-conservative).
    """
    # Simple Kepler + J2 propagator for generating the clean dataset
    # Initial state: circular orbit at 500km altitude
    alt = 500.0 * 1000.0
    r_mag = R_EARTH + alt
    v_mag = np.sqrt(MU_EARTH / r_mag)
    
    pos0 = np.array([r_mag, 0.0, 0.0])
    vel0 = np.array([0.0, v_mag * np.cos(51.6 * np.pi / 180), v_mag * np.sin(51.6 * np.pi / 180)])
    state = np.concatenate([pos0, vel0])
    
    states = []
    derivatives = []
    
    # We will simulate a small constant residual acceleration of [0.01, -0.005, 0.02] m/s^2
    a_residual = np.array([0.01, -0.005, 0.02])
    
    for _ in range(n_steps):
        pos = state[:3]
        vel = state[3:]
        r = np.linalg.norm(pos)
        
        # Keplerian gravity
        a_grav = -MU_EARTH / (r**3) * pos
        
        # J2 perturbation
        x, y, z = pos[0], pos[1], pos[2]
        r2 = r*r
        z2 = z*z
        f2 = 1.5 * 1.08262668e-3 * MU_EARTH * (R_EARTH**2) / (r**5)
        a_j2 = np.array([
            f2 * x * (5.0 * z2 / r2 - 1.0),
            f2 * y * (5.0 * z2 / r2 - 1.0),
            f2 * z * (5.0 * z2 / r2 - 3.0)
        ])
        
        # Total physics acceleration
        a_physics = a_grav + a_j2
        
        # Derivative: [v, a_physics + a_residual]
        d_state = np.concatenate([vel, a_physics + a_residual])
        
        states.append(state.copy())
        derivatives.append(d_state)
        
        # RK4 step to advance state
        k1 = d_state
        
        # Midpoint 1
        st_half1 = state + 0.5 * dt_s * k1
        r_h1 = np.linalg.norm(st_half1[:3])
        a_grav_h1 = -MU_EARTH / (r_h1**3) * st_half1[:3]
        f2_h1 = 1.5 * 1.08262668e-3 * MU_EARTH * (R_EARTH**2) / (r_h1**5)
        a_j2_h1 = np.array([
            f2_h1 * st_half1[0] * (5.0 * st_half1[2]**2 / r_h1**2 - 1.0),
            f2_h1 * st_half1[1] * (5.0 * st_half1[2]**2 / r_h1**2 - 1.0),
            f2_h1 * st_half1[2] * (5.0 * st_half1[2]**2 / r_h1**2 - 3.0)
        ])
        k2 = np.concatenate([st_half1[3:], a_grav_h1 + a_j2_h1 + a_residual])
        
        # Midpoint 2
        st_half2 = state + 0.5 * dt_s * k2
        r_h2 = np.linalg.norm(st_half2[:3])
        a_grav_h2 = -MU_EARTH / (r_h2**3) * st_half2[:3]
        f2_h2 = 1.5 * 1.08262668e-3 * MU_EARTH * (R_EARTH**2) / (r_h2**5)
        a_j2_h2 = np.array([
            f2_h2 * st_half2[0] * (5.0 * st_half2[2]**2 / r_h2**2 - 1.0),
            f2_h2 * st_half2[1] * (5.0 * st_half2[2]**2 / r_h2**2 - 1.0),
            f2_h2 * st_half2[2] * (5.0 * st_half2[2]**2 / r_h2**2 - 3.0)
        ])
        k3 = np.concatenate([st_half2[3:], a_grav_h2 + a_j2_h2 + a_residual])
        
        # End step
        st_end = state + dt_s * k3
        r_e = np.linalg.norm(st_end[:3])
        a_grav_e = -MU_EARTH / (r_e**3) * st_end[:3]
        f2_e = 1.5 * 1.08262668e-3 * MU_EARTH * (R_EARTH**2) / (r_e**5)
        a_j2_e = np.array([
            f2_e * st_end[0] * (5.0 * st_end[2]**2 / r_e**2 - 1.0),
            f2_e * st_end[1] * (5.0 * st_end[2]**2 / r_e**2 - 1.0),
            f2_e * st_end[2] * (5.0 * st_end[2]**2 / r_e**2 - 3.0)
        ])
        k4 = np.concatenate([st_end[3:], a_grav_e + a_j2_e + a_residual])
        
        state += (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    return torch.tensor(np.array(states), dtype=torch.float64), torch.tensor(np.array(derivatives), dtype=torch.float64)


def calculate_energy(state: torch.Tensor) -> torch.Tensor:
    """Calculate the total orbital energy per unit mass: E = 1/2 v^2 - mu/r."""
    pos = state[:, :3]
    vel = state[:, 3:]
    r_mag = torch.norm(pos, dim=-1)
    v2 = torch.sum(vel**2, dim=-1)
    return 0.5 * v2 - MU_EARTH / r_mag


def benchmark_model(model_name: str, vector_field: nn.Module, state_train: torch.Tensor, deriv_train: torch.Tensor, epochs: int = 150) -> dict:
    """Train the model to fit derivatives, then evaluate performance."""
    device = torch.device("cpu")
    vector_field.to(device)
    vector_field.double()  # Force all layers and weights to double precision
    
    # We want to train the model to minimize derivative prediction error
    optimizer = torch.optim.Adam(vector_field.parameters(), lr=0.01)
    
    t_dummy = torch.tensor(0.0, dtype=torch.float64)
    
    print(f"Training {model_name}...")
    t0 = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Evaluate model derivatives
        pred_deriv = vector_field(t_dummy, state_train)
        
        # Match only the acceleration component (last 3 elements) since velocity derivative is fixed to p
        loss = torch.mean((pred_deriv[:, 3:] - deriv_train[:, 3:])**2)
        
        loss.backward()
        optimizer.step()
        
    train_time = time.time() - t0
    final_loss = loss.item()
    print(f"  {model_name} trained in {train_time:.2f}s, final loss: {final_loss:.6e}")
    
    # Evaluate 24-hour integration
    dt = 60.0
    steps_24h = 1440
    t_eval = torch.arange(0.0, steps_24h * dt, dt, dtype=torch.float64)
    state0 = state_train[0:1].clone()
    
    # Run ODE propagation
    t_int0 = time.time()
    with torch.no_grad():
        # RK4 integration
        sol = odeint(
            vector_field,
            state0,  # Shape (1, 6)
            t_eval,
            method="rk4",
            options={"step_size": dt}
        )
    integration_time = time.time() - t_int0
    
    # sol shape is (len(t), 1, 6) -> squeeze out the batch dimension to get (len(t), 6)
    sol = sol[:, 0]
    
    # Calculate drift metrics
    # 1. Position error at the end of 24h against ground-truth
    gt_traj = state_train[:steps_24h]
    pos_err = torch.norm(sol[:, :3] - gt_traj[:, :3], dim=-1)
    final_pos_err_km = pos_err[-1].item() / 1000.0
    
    # 2. Energy drift (Hamiltonian drift) over 24h
    energies = calculate_energy(sol)
    initial_energy = energies[0].item()
    energy_drift = torch.abs(energies - initial_energy)
    max_energy_drift = torch.max(energy_drift).item()
    mean_energy_drift = torch.mean(energy_drift).item()
    
    # 3. Time reversibility check
    # Propagate forward 6 hours (360 steps), then backward 6 hours
    steps_6h = 360
    t_fwd = torch.arange(0.0, steps_6h * dt, dt, dtype=torch.float64)
    t_bwd = torch.arange(steps_6h * dt, 0.0, -dt, dtype=torch.float64)
    
    with torch.no_grad():
        # Fwd
        fwd_sol = odeint(vector_field, state0, t_fwd, method="rk4", options={"step_size": dt})
        # Bwd
        bwd_sol = odeint(vector_field, fwd_sol[-1], t_bwd, method="rk4", options={"step_size": dt})
        
    reversibility_err_m = torch.norm(bwd_sol[-1, 0, :3] - state0[0, :3]).item()
    
    return {
        "model": model_name,
        "train_loss": final_loss,
        "train_time_s": train_time,
        "integration_time_s": integration_time,
        "pos_err_24h_km": final_pos_err_km,
        "max_energy_drift": max_energy_drift,
        "mean_energy_drift": mean_energy_drift,
        "reversibility_err_m": reversibility_err_m,
    }


def main():
    print("=" * 80)
    print("ORBITAL WORLD MODEL BENCHMARK: HNN vs SCALAR POTENTIAL vs PI-NODE")
    print("=" * 80)
    
    # 1. Generate ground truth trajectory of 2000 steps (approx 33 hours)
    print("Generating synthetic perturbed orbit dataset...")
    states, derivatives = generate_benchmark_trajectory(n_steps=2000, dt_s=60.0)
    print(f"Dataset generated: {states.size(0)} steps of position and velocity.")
    
    # 2. Initialize networks
    # MLP hidden dimension is set to 64 for all models to keep parameter count similar
    hidden_dim = 64
    
    # Baseline PI-NODE
    baseline_mlp = ResidualAccelerationNet(hidden_dim=hidden_dim, num_layers=3, dropout=0.0)
    baseline_vf = NeuralODEVectorField(baseline_mlp, bstar=0.0, use_gravity=True, use_j2=True, use_drag=False)
    
    # Scalar Potential Neural ODE
    potential_mlp = DifferentiablePotentialMLP(hidden_dim=hidden_dim)
    potential_vf = ScalarPotentialNeuralODEVectorField(potential_mlp, bstar=0.0, use_gravity=True, use_j2=True, use_drag=False)
    
    # Separable HNN (Momentum and position separated)
    sep_hnn_mlp = DifferentiableHamiltonianMLP(hidden_dim=hidden_dim)
    sep_hnn_vf = HamiltonianNeuralODEVectorField(sep_hnn_mlp, bstar=0.0, use_gravity=True, use_j2=True, use_drag=False, separable=True)
    
    # Coupled HNN (Non-separable)
    coupled_hnn_mlp = DifferentiableHamiltonianMLP(hidden_dim=hidden_dim)
    coupled_hnn_vf = HamiltonianNeuralODEVectorField(coupled_hnn_mlp, bstar=0.0, use_gravity=True, use_j2=True, use_drag=False, separable=False)
    
    # 3. Train and Benchmark
    models_to_test = [
        ("PI-NODE (Baseline MLP)", baseline_vf),
        ("Scalar Potential Model", potential_vf),
        ("Separable HNN", sep_hnn_vf),
        ("Non-Separable HNN (Coupled)", coupled_hnn_vf)
    ]
    
    results = []
    for name, vf in models_to_test:
        res = benchmark_model(name, vf, states, derivatives, epochs=200)
        results.append(res)
        
    # 4. Print Summary Table
    print("\n" + "=" * 95)
    print(f"{'Model Name':<30} | {'Train Loss':<10} | {'24h Pos Err':<12} | {'Max Energy Drift':<16} | {'Reversibility':<13}")
    print("-" * 95)
    
    for r in results:
        print(f"{r['model']:<30} | {r['train_loss']:<10.3e} | {r['pos_err_24h_km']:<9.3f} km | {r['max_energy_drift']:<16.4f} | {r['reversibility_err_m']:<9.3f} m")
    print("=" * 95)
    print("Note: Energy units are in J/kg (specific orbital energy).")
    print("Note: Reversibility measures integration reconstruction error after propagating forward then backward 6 hours.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
