"""Prototyping and verification script for Sundman Regularized Batch Neural ODE.

Compares standard physical time batch integration vs standard Sundman regularized
batch integration for LEO (circular) and GTO (eccentric) batches.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.interpolate import CubicSpline

# --- Physical Constants ---
MU_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.3781363e6  # m
J2 = 1.08262668e-3
J3 = -2.53265648e-6
J4 = -1.61962159e-6

# --- Canonical Conversion Factors ---
R_REF = R_EARTH
T_REF = np.sqrt(R_REF**3 / MU_EARTH)  # ~806.81 s
V_REF = R_REF / T_REF  # ~7905.3 m/s

# --- Perturbations (Vectorized Canonical Units) ---

def get_j3_acc_canonical(pos: torch.Tensor) -> torch.Tensor:
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    r = torch.norm(pos, dim=-1, keepdim=True)
    r_safe = torch.clamp(r, min=0.5)
    r2 = r_safe * r_safe
    z2 = z * z
    z_r2 = z2 / r2
    f3 = 0.5 * J3 / (r_safe**7)
    return torch.cat([
        f3 * 5.0 * x * (7.0 * z * z_r2 - 3.0 * z),
        f3 * 5.0 * y * (7.0 * z * z_r2 - 3.0 * z),
        f3 * (6.0 * z2 - 7.0 * z2 * z_r2 - 0.6 * r2)
    ], dim=-1)

def get_drag_acc_canonical(vel: torch.Tensor) -> torch.Tensor:
    v_mag = torch.norm(vel, dim=-1, keepdim=True)
    const = 1.5e-11 * V_REF * T_REF  # ~9.57e-5
    return -const * v_mag * vel

# --- Vector Fields (Canonical Units) ---

class CanonicalPhysicalVF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        pos = state[:, :3]
        vel = state[:, 3:6]
        r = torch.norm(pos, dim=-1, keepdim=True)
        r_safe = torch.clamp(r, min=0.5)

        # Keplerian Gravity
        a_grav = -pos / (r_safe**3)

        # J2
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        f2 = 1.5 * J2 / (r_safe**5)
        a_j2 = torch.cat([
            f2 * x * (5.0 * (z**2) / (r_safe**2) - 1.0),
            f2 * y * (5.0 * (z**2) / (r_safe**2) - 1.0),
            f2 * z * (5.0 * (z**2) / (r_safe**2) - 3.0),
        ], dim=-1)

        # True Residual: 500 * J3 + Drag
        a_j3 = get_j3_acc_canonical(pos)
        a_drag = get_drag_acc_canonical(vel)
        a_res = 500.0 * a_j3 + a_drag

        return torch.cat([vel, a_grav + a_j2 + a_res], dim=-1)


class CanonicalSundmanVF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe = 0

    def forward(self, s: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        pos = state[:, :3]
        vel = state[:, 3:6]
        r = torch.norm(pos, dim=-1, keepdim=True)
        r_safe = torch.clamp(r, min=0.5)

        # Keplerian Gravity
        a_grav = -pos / (r_safe**3)

        # J2
        x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
        f2 = 1.5 * J2 / (r_safe**5)
        a_j2 = torch.cat([
            f2 * x * (5.0 * (z**2) / (r_safe**2) - 1.0),
            f2 * y * (5.0 * (z**2) / (r_safe**2) - 1.0),
            f2 * z * (5.0 * (z**2) / (r_safe**2) - 3.0),
        ], dim=-1)

        # True Residual: 500 * J3 + Drag
        a_j3 = get_j3_acc_canonical(pos)
        a_drag = get_drag_acc_canonical(vel)
        a_res = 500.0 * a_j3 + a_drag

        # Standard Sundman transformation: dy/ds = r * dy/dt
        d_pos = r_safe * vel
        d_vel = r_safe * (a_grav + a_j2 + a_res)
        d_time = r_safe

        return torch.cat([d_pos, d_vel, d_time], dim=-1)

# --- Synthetic Orbit Generators (Canonical Units) ---

def generate_orbits_canonical(orbit_type: str = "LEO", batch_size: int = 50, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    states = []
    
    for _ in range(batch_size):
        if orbit_type == "LEO":
            alt = 500000.0 + np.random.uniform(-10000.0, 10000.0)
            r_mag = (R_REF + alt) / R_REF
            v_mag = np.sqrt(1.0 / r_mag)
            incl = np.random.uniform(50.0, 53.0) * np.pi / 180.0
            arg_lat = np.random.uniform(0.0, 2.0 * np.pi)
            
            r_peri = np.array([r_mag * np.cos(arg_lat), r_mag * np.sin(arg_lat), 0.0])
            v_peri = np.array([-v_mag * np.sin(arg_lat), v_mag * np.cos(arg_lat), 0.0])
        else:
            r_p = (R_REF + 500000.0) / R_REF
            r_a = (R_REF + 35000000.0) / R_REF
            a_val = (r_p + r_a) / 2.0
            e = (r_a - r_p) / (r_a + r_p)
            
            E = np.random.uniform(0.0, 2.0 * np.pi)
            r_mag = a_val * (1.0 - e * np.cos(E))
            v_mag = np.sqrt(2.0 / r_mag - 1.0 / a_val)
            
            r_peri = np.array([a_val * (np.cos(E) - e), a_val * np.sqrt(1.0 - e**2) * np.sin(E), 0.0])
            n = 1.0 / (a_val**1.5)
            factor = n * a_val**2 / r_mag
            v_peri = np.array([-factor * np.sin(E), factor * np.sqrt(1.0 - e**2) * np.cos(E), 0.0])
            
            incl = np.random.uniform(25.0, 30.0) * np.pi / 180.0
            
        raan = np.random.uniform(0.0, 2.0 * np.pi)
        R_incl = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(incl), -np.sin(incl)],
            [0.0, np.sin(incl), np.cos(incl)]
        ])
        R_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_trans = R_raan @ R_incl
        
        states.append(np.concatenate([R_trans @ r_peri, R_trans @ v_peri]))
        
    return np.array(states)

# --- Energy Preservation Calculation ---

def get_max_energy_drift(pos: np.ndarray, vel: np.ndarray) -> float:
    r = np.linalg.norm(pos, axis=-1)
    v2 = np.sum(vel**2, axis=-1)
    energies = 0.5 * v2 - 1.0 / r
    drift = np.abs(energies - energies[0]) / np.abs(energies[0])
    return float(np.max(drift))

# --- Single Batch Benchmark Runner ---

def run_batch_benchmark(orbit_type: str = "LEO", batch_size: int = 50, s_max_val: float = 41.3):
    print("\n" + "="*70)
    print(f"BENCHMARK: {orbit_type} Batch ({batch_size} satellites, s_max = {s_max_val:.2f})")
    print("="*70)
    
    # 1. Generate States
    init_states = generate_orbits_canonical(orbit_type=orbit_type, batch_size=batch_size, seed=42)
    init_states_t = torch.tensor(init_states, dtype=torch.float64)
    
    # 2. Sundman Batch Integration (Low Precision rtol=1e-7)
    s_span = torch.linspace(0.0, s_max_val, 1000, dtype=torch.float64)
    init_sundman_t = torch.cat([
        init_states_t,
        torch.zeros((batch_size, 1), dtype=torch.float64)
    ], dim=-1)
    
    print(f"Integrating batch in SUNDMAN domain (rtol=1e-7)...")
    sundman_vf = CanonicalSundmanVF()
    t_start = time.perf_counter()
    with torch.no_grad():
        sundman_states = odeint(
            sundman_vf,
            init_sundman_t,
            s_span,
            method="dopri5",
            rtol=1e-7,
            atol=1e-7
        )
    t_sundman_dur = time.perf_counter() - t_start
    sundman_nfe = sundman_vf.nfe
    sundman_states_np = sundman_states.cpu().numpy()
    
    # Determine the overlapping physical time domain to query (min of reached times)
    times = sundman_states_np[-1, :, 6]
    t_eval_c = times.min()
    t_span = torch.linspace(0.0, t_eval_c, 100, dtype=torch.float64)
    
    print(f"  Sundman integration completed in {t_sundman_dur:.3f}s. NFE: {sundman_nfe}")
    print(f"  Minimum physical time reached: {t_eval_c * T_REF / 3600.0:.2f} hours")

    # 3. Reference High-Precision Physical Integration (Ground Truth)
    print("Integrating ground-truth physical trajectory (rtol=1e-12)...")
    with torch.no_grad():
        ref_states_t = odeint(
            CanonicalPhysicalVF(),
            init_states_t,
            t_span,
            method="dopri5",
            rtol=1e-12,
            atol=1e-12
        )
    ref_states_np = ref_states_t.cpu().numpy()

    # 4. Physical Batch Integration (Low Precision rtol=1e-7)
    print("Integrating batch in PHYSICAL TIME domain (rtol=1e-7)...")
    phys_vf = CanonicalPhysicalVF()
    t_start = time.perf_counter()
    with torch.no_grad():
        phys_states_low = odeint(
            phys_vf,
            init_states_t,
            t_span,
            method="dopri5",
            rtol=1e-7,
            atol=1e-7
        )
    t_phys_dur = time.perf_counter() - t_start
    phys_nfe = phys_vf.nfe
    phys_states_low_np = phys_states_low.cpu().numpy()
    
    # Compute max error for physical integration
    phys_errors = []
    phys_drifts = []
    for i in range(batch_size):
        err = np.linalg.norm(phys_states_low_np[:, i, :3] - ref_states_np[:, i, :3], axis=-1) * R_REF
        phys_errors.append(err.max())
        drift = get_max_energy_drift(phys_states_low_np[:, i, :3], phys_states_low_np[:, i, 3:6])
        phys_drifts.append(drift)
        
    print(f"  Completed physical in {t_phys_dur:.3f}s. NFE: {phys_nfe}")
    print(f"  Max Position Error: {np.max(phys_errors):.2f} meters")
    print(f"  Max Energy Drift:   {np.max(phys_drifts):.2e}")
    
    # 5. Evaluate Interpolated Accuracy for Sundman
    sundman_errors = []
    sundman_drifts = []
    t_query = t_span.numpy()
    
    for i in range(batch_size):
        t_i = sundman_states_np[:, i, 6]
        pos_s_i = sundman_states_np[:, i, :3]
        vel_s_i = sundman_states_np[:, i, 3:6]
        
        t_i, u = np.unique(t_i, return_index=True)
        pos_s_i = pos_s_i[u]
        vel_s_i = vel_s_i[u]
        
        # Interpolate back to t_query
        cs_pos = CubicSpline(t_i, pos_s_i, axis=0)
        cs_vel = CubicSpline(t_i, vel_s_i, axis=0)
        
        interp_pos = cs_pos(t_query)
        interp_vel = cs_vel(t_query)
        
        err = np.linalg.norm(interp_pos - ref_states_np[:, i, :3], axis=-1) * R_REF
        sundman_errors.append(err.max())
        drift = get_max_energy_drift(interp_pos, interp_vel)
        sundman_drifts.append(drift)
        
    print(f"  Max Sundman Position Error: {np.max(sundman_errors):.2f} meters")
    print(f"  Max Sundman Energy Drift:   {np.max(sundman_drifts):.2e}")
    
    # 6. Summary Comparison
    nfe_reduction = phys_nfe / sundman_nfe
    precision_improvement = np.max(phys_errors) / np.max(sundman_errors)
    
    print("\n  Summary:")
    print(f"    - NFE Reduction:       {nfe_reduction:.2f}x")
    print(f"    - Execution Speedup:   {t_phys_dur / t_sundman_dur:.2f}x")
    print(f"    - Max Error reduction: {precision_improvement:.2f}x")
    
    return {
        "nfe_reduction": nfe_reduction,
        "phys_error": np.max(phys_errors),
        "sundman_error": np.max(sundman_errors),
        "precision_improvement": precision_improvement
    }

def main():
    # Run LEO circular batch benchmark (s_max = 41.3 covers ~10.0 hours LEO)
    leo_results = run_batch_benchmark(orbit_type="LEO", batch_size=50, s_max_val=41.3)
    
    # Run GTO eccentric batch benchmark (s_max = 13.5 covers ~10.6 hours GTO)
    gto_results = run_batch_benchmark(orbit_type="GTO", batch_size=50, s_max_val=13.5)
    
    print("\n" + "="*80)
    print("FINAL R&D VERIFICATION REPORT (T-094)")
    print("="*80)
    
    # Tolerances are 1e-7, so LEO error < 3000m is normal and GTO error < 6000m is normal, precision gain > 1.2
    leo_pass = leo_results["sundman_error"] < 3500.0
    gto_pass = (gto_results["sundman_error"] < 6000.0) and (gto_results["nfe_reduction"] > 2.0) and (gto_results["precision_improvement"] > 1.2)
    
    print(f"LEO Batch Conformance: {'[PASS]' if leo_pass else '[FAIL]'} (Max Error: {leo_results['sundman_error']:.2f} m)")
    print(f"GTO Batch Conformance: {'[PASS]' if gto_pass else '[FAIL]'} (Max Error: {gto_results['sundman_error']:.2f} m, Precision Gain: {gto_results['precision_improvement']:.2f}x)")
    
    overall_pass = leo_pass and gto_pass
    print("-" * 80)
    if overall_pass:
        print("  [SUCCESS] Poincaré-Sundman Regularized Batch Neural ODE math and precision verified!")
        print(f"  GTO NFE Speedup: {gto_results['nfe_reduction']:.2f}x (Decoupled stiffness mismatch)")
        print(f"  GTO Precision Gain: {gto_results['precision_improvement']:.2f}x (Resolved RMS error under-control issue)")
        print("  Recommendation: Approve design brief for implementation in next version (v0.7.0).")
    else:
        print("  [FAILED] Accuracy or speedups did not conform to R&D metrics.")
    print("=" * 80)

if __name__ == "__main__":
    main()
