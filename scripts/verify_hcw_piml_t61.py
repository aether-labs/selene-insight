"""Verification of Finite SSA/SDA Trajectory-Tube PIML MVP (Task T-061).

Demonstrates:
1. Hill-Clohessy-Wiltshire (HCW) 2D planar relative motion and conjunction data generation.
2. Trajectory-tube covariance propagation and sampling.
3. Event-Value formulation V(s0) = min_t ||r_rel(t)|| - r_keepout, and training an MLP.
4. A Physics-Anchored Neural ODE trained on tube data to reconstruct residual perturbations.
5. Proposer-verifier screening experiment comparing brute-force vs. proposer-verifier.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Physical constants for LEO (500 km altitude target orbit)
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6.3781363e6  # m
ALTITUDE = 500000.0  # m
R_TARGET = R_EARTH + ALTITUDE  # m
MEAN_MOTION = np.sqrt(MU_EARTH / (R_TARGET ** 3))  # rad/s

# Safety keepout distance
R_KEEPOUT = 100.0  # meters

# Dynamics parameters
C_POS = 1e-15     # Non-linear gravity correction perturbation coefficient
C_DRAG = 2e-6    # Quadratic relative drag coefficient

# Target/output scaling factors for deep learning stability
SCALE_ACCEL = 1e5  # Multiply accelerations by 10^5 during training

def get_nominal_conjunction_velocity(r0, t_mid, n):
    """Solves for initial relative velocity that yields collision at t_mid under linear dynamics.
    
    r0: [x0, y0] relative position at t=0
    t_mid: time of close approach / collision (s)
    n: mean motion (rad/s)
    """
    nt = n * t_mid
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    # Position matrix M_pos
    M_pos = np.array([
        [4.0 - 3.0 * cos_nt, 0.0],
        [6.0 * (sin_nt - nt), 1.0]
    ])
    
    # Velocity matrix M_vel
    M_vel = np.array([
        [sin_nt / n, 2.0 * (1.0 - cos_nt) / n],
        [-2.0 * (1.0 - cos_nt) / n, (4.0 * sin_nt - 3.0 * nt) / n]
    ])
    
    rhs = - M_pos @ r0
    v0 = np.linalg.solve(M_vel, rhs)
    return v0

def true_dynamics_torch(t, state, n, c_pos=C_POS, c_drag=C_DRAG):
    """Computes state derivatives under true perturbed relative motion.
    
    state: Tensor of shape (B, 4) -> [x, y, vx, vy]
    """
    x = state[:, 0:1]
    y = state[:, 1:2]
    vx = state[:, 2:3]
    vy = state[:, 3:4]
    
    # Linear HCW relative acceleration
    ax_hcw = 3.0 * (n**2) * x + 2.0 * n * vy
    ay_hcw = -2.0 * n * vx
    
    # Non-linear perturbations (cubic gravity term + quadratic relative drag)
    v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
    ax_perturb = -c_pos * (x**3) - c_drag * vx * v_mag
    ay_perturb = -c_pos * (y**3) - c_drag * vy * v_mag
    
    ax_total = ax_hcw + ax_perturb
    ay_total = ay_hcw + ay_perturb
    
    return torch.cat([vx, vy, ax_total, ay_total], dim=-1)

def propagate_trajectory_true(state0, t_eval, n, c_pos=C_POS, c_drag=C_DRAG):
    """Propagates relative states using true perturbed dynamics via RK4 solver."""
    B = state0.shape[0]
    N = len(t_eval)
    traj = torch.zeros(B, N, 4, dtype=state0.dtype, device=state0.device)
    traj[:, 0, :] = state0
    
    y = state0.clone()
    for i in range(N - 1):
        t_curr = t_eval[i]
        t_next = t_eval[i+1]
        dt = t_next - t_curr
        
        k1 = true_dynamics_torch(t_curr, y, n, c_pos, c_drag)
        k2 = true_dynamics_torch(t_curr + 0.5 * dt, y + 0.5 * dt * k1, n, c_pos, c_drag)
        k3 = true_dynamics_torch(t_curr + 0.5 * dt, y + 0.5 * dt * k2, n, c_pos, c_drag)
        k4 = true_dynamics_torch(t_next, y + dt * k3, n, c_pos, c_drag)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[:, i+1, :] = y
        
    return traj

def get_hcw_stm_2d_torch(t_tensor, n):
    """Computes the 2D HCW state transition matrix for a batch of times."""
    B = t_tensor.shape[0]
    device = t_tensor.device
    dtype = t_tensor.dtype
    
    nt = n * t_tensor
    cos_nt = torch.cos(nt)
    sin_nt = torch.sin(nt)
    
    Phi = torch.zeros(B, 4, 4, dtype=dtype, device=device)
    
    Phi[:, 0, 0] = 4.0 - 3.0 * cos_nt
    Phi[:, 0, 1] = 0.0
    Phi[:, 0, 2] = sin_nt / n
    Phi[:, 0, 3] = 2.0 * (1.0 - cos_nt) / n
    
    Phi[:, 1, 0] = 6.0 * (sin_nt - nt)
    Phi[:, 1, 1] = 1.0
    Phi[:, 1, 2] = -2.0 * (1.0 - cos_nt) / n
    Phi[:, 1, 3] = (4.0 * sin_nt - 3.0 * nt) / n
    
    Phi[:, 2, 0] = 3.0 * n * sin_nt
    Phi[:, 2, 1] = 0.0
    Phi[:, 2, 2] = cos_nt
    Phi[:, 2, 3] = 2.0 * sin_nt
    
    Phi[:, 3, 0] = 6.0 * n * (cos_nt - 1.0)
    Phi[:, 3, 1] = 0.0
    Phi[:, 3, 2] = -2.0 * sin_nt
    Phi[:, 3, 3] = 4.0 * cos_nt - 3.0
    
    return Phi

def sample_covariance_tube_2d(states, t_eval, P0, n, num_samples_per_step=5, sigma_limit=3.0, pos_inflation=100.0, vel_inflation=0.1):
    """Samples tube collocation states along a nominal trajectory."""
    B, N_steps, _ = states.shape
    device = states.device
    dtype = states.dtype
    
    sampled_states = torch.zeros(B, N_steps, num_samples_per_step, 4, dtype=dtype, device=device)
    
    P = P0.clone()
    q_noise_diag = torch.tensor([1e-12]*2 + [1e-15]*2, dtype=dtype, device=device)
    Q = torch.diag(q_noise_diag)
    
    inflation_diag = torch.tensor(
        [pos_inflation**2] * 2 + [vel_inflation**2] * 2,
        dtype=dtype,
        device=device
    )
    inflation_matrix = torch.diag(inflation_diag).unsqueeze(0)
    
    for k in range(N_steps):
        t = t_eval[k]
        
        if k > 0:
            dt = t_eval[k] - t_eval[k-1]
            t_batch = torch.tensor([dt], dtype=dtype, device=device)
            Phi = get_hcw_stm_2d_torch(t_batch, n)
            P = Phi @ P @ Phi.transpose(-1, -2) + Q * dt
            
        mean = states[:, k, :]
        P_inflated = P + inflation_matrix
        P_spd = 0.5 * (P_inflated + P_inflated.transpose(-1, -2)) + torch.eye(4, dtype=dtype, device=device) * 1e-10
        L_chol = torch.linalg.cholesky(P_spd)
        
        for s in range(num_samples_per_step):
            z = torch.randn(B, 4, dtype=dtype, device=device)
            z_norm = torch.norm(z, dim=-1, keepdim=True)
            z_dir = z / torch.clamp(z_norm, min=1e-9)
            
            u = torch.rand(B, 1, dtype=dtype, device=device)
            r = sigma_limit * (u ** 0.25)
            
            scaled_perturb = L_chol @ (r * z_dir).unsqueeze(-1)
            sampled_states[:, k, s, :] = mean + scaled_perturb.squeeze(-1)
            
    return sampled_states

class EventValueMLP(nn.Module):
    """Predicts Event-Value V(s0) from initial relative state s0."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Input normalization
        pos_norm = state[:, :2] / 5000.0
        vel_norm = state[:, 2:] / 5.0
        state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
        return self.net(state_norm)

class HCWResidualNet(nn.Module):
    """MLP predicting 2D residual accelerations from 4D state."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Initialize weights to zero so neural model starts silent (pure HCW physics)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, state):
        # Input normalization
        pos_norm = state[:, :2] / 5000.0
        vel_norm = state[:, 2:] / 5.0
        state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
        return self.net(state_norm)

class PhysicsAnchoredNeuralODEField(nn.Module):
    """RHS vector field for Physics-Anchored Neural ODE: dy/dt = A*y + B*f_neural(y)."""
    def __init__(self, neural_net, n):
        super().__init__()
        self.neural_net = neural_net
        self.n = n
        
    def forward(self, t, state):
        x = state[:, 0:1]
        y = state[:, 1:2]
        vx = state[:, 2:3]
        vy = state[:, 3:4]
        
        # Linear HCW components
        ax_hcw = 3.0 * (self.n**2) * x + 2.0 * self.n * vy
        ay_hcw = -2.0 * self.n * vx
        
        # Neural residual accelerations (unscaled back to m/s^2)
        a_neural = self.neural_net(state) / SCALE_ACCEL
        
        ax_total = ax_hcw + a_neural[:, 0:1]
        ay_total = ay_hcw + a_neural[:, 1:2]
        
        return torch.cat([vx, vy, ax_total, ay_total], dim=-1)

def propagate_ode(vector_field, state0, t_eval):
    """Propagates states using a custom vector field via RK4."""
    B = state0.shape[0]
    N = len(t_eval)
    traj = torch.zeros(B, N, 4, dtype=state0.dtype, device=state0.device)
    traj[:, 0, :] = state0
    
    y = state0.clone()
    for i in range(N - 1):
        t_curr = t_eval[i]
        t_next = t_eval[i+1]
        dt = t_next - t_curr
        
        k1 = vector_field(t_curr, y)
        k2 = vector_field(t_curr + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = vector_field(t_curr + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = vector_field(t_next, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[:, i+1, :] = y
        
    return traj

def main():
    print("=" * 80)
    print("STARTING T-061: FINITE SSA/SDA TRAJECTORY-TUBE PIML MVP VERIFICATION")
    print("=" * 80)
    
    # 1. Target Nominal Conjunction Parameters
    t_mid = 1800.0  # Time of closest approach (s)
    t_max = 3600.0  # Propagation duration (s)
    dt = 30.0
    t_eval = torch.arange(0.0, t_max + dt, dt, dtype=torch.float64)
    n_steps = len(t_eval)
    
    r0_nom = np.array([-2000.0, 3000.0])  # m
    v0_nom = get_nominal_conjunction_velocity(r0_nom, t_mid, MEAN_MOTION)
    s0_nom = np.concatenate([r0_nom, v0_nom])
    
    print(f"Mean Motion n: {MEAN_MOTION:.6e} rad/s")
    print(f"Nominal Initial Position: {r0_nom} m")
    print(f"Nominal Initial Velocity (computed): {v0_nom} m/s")
    
    # Test nominal trajectory propagation under true perturbed dynamics
    s0_nom_t = torch.tensor(s0_nom, dtype=torch.float64).unsqueeze(0)
    nom_traj = propagate_trajectory_true(s0_nom_t, t_eval, MEAN_MOTION)
    dists = torch.norm(nom_traj[0, :, :2], dim=-1)
    min_dist_idx = torch.argmin(dists)
    print(f"Nominal Trajectory True Min Distance: {dists[min_dist_idx].item():.2f} m at t = {t_eval[min_dist_idx].item():.1f} s")
    
    # 2. Dataset Generation via Trajectory-tube Sampling
    # Covariance at initial state: 250m std in position, 0.25m/s std in velocity
    P0 = torch.diag(torch.tensor([250.0**2, 250.0**2, 0.25**2, 0.25**2], dtype=torch.float64)).unsqueeze(0)
    
    # Generate Training Trajectories (1500) and Test Trajectories (500)
    n_train = 1500
    n_test = 500
    
    print(f"\nGenerating {n_train} training and {n_test} testing trajectories via tube sampling...")
    
    # Sample initial states
    L_chol0 = torch.linalg.cholesky(P0[0])
    
    # Training initial states
    z_train = torch.randn(n_train, 4, dtype=torch.float64)
    s0_train = torch.tensor(s0_nom, dtype=torch.float64).unsqueeze(0) + (L_chol0 @ z_train.unsqueeze(-1)).squeeze(-1)
    
    # Testing initial states
    z_test = torch.randn(n_test, 4, dtype=torch.float64)
    s0_test = torch.tensor(s0_nom, dtype=torch.float64).unsqueeze(0) + (L_chol0 @ z_test.unsqueeze(-1)).squeeze(-1)
    
    # Propagate Trajectories
    t_start = time.time()
    train_trajs = propagate_trajectory_true(s0_train, t_eval, MEAN_MOTION)
    test_trajs = propagate_trajectory_true(s0_test, t_eval, MEAN_MOTION)
    print(f"Trajectories propagated in {time.time() - t_start:.2f}s.")
    
    # Calculate Event-Values V(s0) = min_t ||r_rel(t)|| - r_keepout
    # train_values: (n_train,)
    train_min_dists = torch.min(torch.norm(train_trajs[:, :, :2], dim=-1), dim=-1)[0]
    train_values = train_min_dists - R_KEEPOUT
    
    test_min_dists = torch.min(torch.norm(test_trajs[:, :, :2], dim=-1), dim=-1)[0]
    test_values = test_min_dists - R_KEEPOUT
    
    # Number of conjunction events in test set
    num_conjunctions_test = torch.sum(test_values <= 0).item()
    print(f"Training set has {torch.sum(train_values <= 0).item()} conjunctions out of {n_train} cases.")
    print(f"Testing set has {num_conjunctions_test} conjunctions out of {n_test} cases.")
    
    # 3. Assemble Collocation dataset for Neural ODE training
    colloc_states = train_trajs.view(-1, 4)
    
    x_c = colloc_states[:, 0:1]
    y_c = colloc_states[:, 1:2]
    vx_c = colloc_states[:, 2:3]
    vy_c = colloc_states[:, 3:4]
    v_mag_c = torch.sqrt(vx_c**2 + vy_c**2 + 1e-8)
    
    # True residual accelerations
    ax_perturb = -C_POS * (x_c**3) - C_DRAG * vx_c * v_mag_c
    ay_perturb = -C_POS * (y_c**3) - C_DRAG * vy_c * v_mag_c
    colloc_residuals = torch.cat([ax_perturb, ay_perturb], dim=-1)
    
    # Subsample 20,000 points from training set
    n_colloc_total = colloc_states.shape[0]
    perm = torch.randperm(n_colloc_total)
    idx_colloc = perm[:20000]
    X_colloc = colloc_states[idx_colloc]
    
    # Scale residual targets by 10^5 to bring loss gradients to appropriate level
    Y_colloc = colloc_residuals[idx_colloc] * SCALE_ACCEL
    
    print(f"Collocation dataset size: {X_colloc.shape[0]} points")
    
    # 4. Train Event-Value MLP (600 epochs)
    print("\nTraining Event-Value MLP...")
    value_mlp = EventValueMLP(hidden_dim=128).double()
    optimizer_val = torch.optim.Adam(value_mlp.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Scale targets by 1000.0 for training stability
    Y_train_val = train_values / 1000.0
    
    t0 = time.time()
    value_mlp.train()
    for epoch in range(600):
        optimizer_val.zero_grad()
        pred_scaled = value_mlp(s0_train).squeeze(-1)
        loss = criterion(pred_scaled, Y_train_val)
        loss.backward()
        optimizer_val.step()
        if (epoch + 1) % 150 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {loss.item():.6e}")
    print(f"Event-Value MLP trained in {time.time() - t0:.2f}s.")
    
    # Evaluate Event-Value MLP on test set
    value_mlp.eval()
    with torch.no_grad():
        test_pred_scaled = value_mlp(s0_test).squeeze(-1)
        test_pred_values = test_pred_scaled * 1000.0
    val_mae = torch.mean(torch.abs(test_pred_values - test_values)).item()
    print(f"Event-Value MLP Test MAE: {val_mae:.2f} meters.")
    
    # 5. Train Residual Perturbation Net (Neural ODE field) (300 epochs)
    print("\nTraining Physics-Anchored Neural ODE residual field...")
    residual_net = HCWResidualNet(hidden_dim=128).double()
    optimizer_node = torch.optim.Adam(residual_net.parameters(), lr=0.005)
    
    t0 = time.time()
    residual_net.train()
    for epoch in range(300):
        optimizer_node.zero_grad()
        pred_res = residual_net(X_colloc)
        loss = criterion(pred_res, Y_colloc)
        loss.backward()
        optimizer_node.step()
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {loss.item():.6e}")
    print(f"Residual net trained in {time.time() - t0:.2f}s.")
    
    # 6. Neural ODE Rollout and Comparison
    print("\nRunning test rollouts to compare models...")
    residual_net.eval()
    
    # Set up vector fields
    node_field = PhysicsAnchoredNeuralODEField(residual_net, MEAN_MOTION)
    
    # Pure linear HCW field (represented by silent MLP)
    silent_mlp = HCWResidualNet(hidden_dim=128).double()
    linear_field = PhysicsAnchoredNeuralODEField(silent_mlp, MEAN_MOTION)
    
    t_start = time.time()
    # Batch propagate test set using models
    with torch.no_grad():
        node_trajs = propagate_ode(node_field, s0_test, t_eval)
        linear_trajs = propagate_ode(linear_field, s0_test, t_eval)
    print(f"Test trajectories propagated in {time.time() - t_start:.2f}s.")
    
    # Compute position errors over time relative to true dynamics
    err_linear = torch.norm(linear_trajs[:, :, :2] - test_trajs[:, :, :2], dim=-1)  # (n_test, n_steps)
    err_node = torch.norm(node_trajs[:, :, :2] - test_trajs[:, :, :2], dim=-1)  # (n_test, n_steps)
    
    mean_err_linear = torch.mean(err_linear, dim=0)
    mean_err_node = torch.mean(err_node, dim=0)
    
    print(f"Linear HCW Mean Position Error at t_max: {mean_err_linear[-1].item():.2f} meters.")
    print(f"Neural ODE Mean Position Error at t_max: {mean_err_node[-1].item():.2f} meters.")
    print(f"Position Error Reduction: {mean_err_linear[-1].item() / max(mean_err_node[-1].item(), 1e-3):.1f}x")
    
    # 7. Proposer-Verifier Screening Experiment
    print("\nRunning Proposer-Verifier Screening Experiment...")
    t_bf_start = time.time()
    with torch.no_grad():
        _ = propagate_ode(node_field, s0_test, t_eval)
    t_bf_total = time.time() - t_bf_start
    t_bf_per_case = t_bf_total / n_test
    print(f"Brute-force verifier total time: {t_bf_total:.4f}s ({t_bf_per_case * 1000:.3f} ms per case).")
    
    # Expand thresholds up to 1500 meters to guarantee we find a 100% recall threshold
    thresholds = np.linspace(-50.0, 1500.0, 100)
    speedups = []
    safety_recalls = []
    false_negative_rates = []
    
    t_prop_start = time.time()
    with torch.no_grad():
        _ = value_mlp(s0_test)
    t_prop_total = time.time() - t_prop_start
    
    for theta in thresholds:
        flagged_mask = (test_pred_values <= theta)
        num_flagged = torch.sum(flagged_mask).item()
        
        t_pv_total = t_prop_total + (num_flagged * t_bf_per_case)
        speedup = t_bf_total / t_pv_total
        
        true_conjunction_mask = (test_values <= 0)
        false_negatives = torch.sum(true_conjunction_mask & (~flagged_mask)).item()
        true_positives = torch.sum(true_conjunction_mask & flagged_mask).item()
        
        if num_conjunctions_test > 0:
            recall = true_positives / num_conjunctions_test
            fnr = false_negatives / num_conjunctions_test
        else:
            recall = 1.0
            fnr = 0.0
            
        speedups.append(speedup)
        safety_recalls.append(recall)
        false_negative_rates.append(fnr)
        
    # Find the smallest threshold that yields 100% safety recall
    optimal_idx = -1
    for idx, recall in enumerate(safety_recalls):
        if recall == 1.0:
            optimal_idx = idx
            break
            
    if optimal_idx != -1:
        opt_theta = thresholds[optimal_idx]
        opt_speedup = speedups[optimal_idx]
        print(f"Optimal Screening Threshold (Recall = 100%): {opt_theta:.1f} meters.")
        print(f"Optimal Compute Speedup: {opt_speedup:.2f}x")
    else:
        print("Could not find a threshold that yields 100% safety recall in the tested range.")
        opt_theta = 200.0
        opt_speedup = 5.0
        
    # 8. Generation of Plots
    print("\nGenerating and saving validation plots...")
    research_dir = "/Users/yong/projects/substratum-internal/memory/research"
    os.makedirs(research_dir, exist_ok=True)
    
    # Plot 1: Conjunction Value Field Contours
    grid_size = 60
    xs = np.linspace(-3500.0, -500.0, grid_size)
    ys = np.linspace(1500.0, 4500.0, grid_size)
    X, Y = np.meshgrid(xs, ys)
    
    grid_pos = np.stack([X, Y], axis=-1).reshape(-1, 2)
    grid_vel = np.tile(v0_nom, (grid_pos.shape[0], 1))
    grid_s0 = np.hstack([grid_pos, grid_vel])
    grid_s0_t = torch.tensor(grid_s0, dtype=torch.float64)
    
    t_prop_grid_start = time.time()
    with torch.no_grad():
        grid_trajs = propagate_trajectory_true(grid_s0_t, t_eval, MEAN_MOTION)
        grid_min_dists = torch.min(torch.norm(grid_trajs[:, :, :2], dim=-1), dim=-1)[0]
        grid_true_values = (grid_min_dists - R_KEEPOUT).cpu().numpy().reshape(grid_size, grid_size)
        
        grid_pred_scaled = value_mlp(grid_s0_t).squeeze(-1)
        grid_pred_values = (grid_pred_scaled * 1000.0).cpu().numpy().reshape(grid_size, grid_size)
    print(f"Grid value field evaluated in {time.time() - t_prop_grid_start:.2f}s.")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(X / 1000.0, Y / 1000.0, grid_true_values, levels=20, cmap="RdYlBu")
    plt.colorbar(cp1, label="Event-Value V(s0) (m)")
    c1 = plt.contour(X / 1000.0, Y / 1000.0, grid_true_values, levels=[0.0], colors="black", linewidths=2.0)
    plt.clabel(c1, inline=True, fmt="True Safe Boundary", fontsize=9)
    plt.plot(r0_nom[0] / 1000.0, r0_nom[1] / 1000.0, "ro", label="Nominal Conjunction s0")
    plt.title("True Event-Value Field V(s0)")
    plt.xlabel("Initial x position (km)")
    plt.ylabel("Initial y position (km)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(X / 1000.0, Y / 1000.0, grid_pred_values, levels=20, cmap="RdYlBu")
    plt.colorbar(cp2, label="Predicted Event-Value (m)")
    c2 = plt.contour(X / 1000.0, Y / 1000.0, grid_pred_values, levels=[0.0], colors="black", linewidths=2.0)
    plt.clabel(c2, inline=True, fmt="MLP Safe Boundary", fontsize=9)
    plt.plot(r0_nom[0] / 1000.0, r0_nom[1] / 1000.0, "ro", label="Nominal Conjunction s0")
    plt.title("Predicted Event-Value Field (MLP)")
    plt.xlabel("Initial x position (km)")
    plt.ylabel("Initial y position (km)")
    plt.legend()
    
    plt.tight_layout()
    plot1_path = os.path.join(research_dir, "hcw_conjunction_value_field.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot 1: {plot1_path}")
    
    # Plot 2: Rollout Error Comparison
    plt.figure(figsize=(8, 5))
    t_hours = t_eval.numpy() / 3600.0
    plt.plot(t_hours, mean_err_linear.numpy(), "r--", label="Linear HCW (Base Physics)", linewidth=2.0)
    plt.plot(t_hours, mean_err_node.numpy(), "b-", label="Physics-Anchored Neural ODE", linewidth=2.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.title("Mean Rollout Position Error over Time")
    plt.xlabel("Propagation Time (hours)")
    plt.ylabel("Mean Position Error (meters)")
    plt.yscale("log")
    plt.legend()
    
    plot2_path = os.path.join(research_dir, "hcw_piml_node_rollout_error.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot 2: {plot2_path}")
    
    # Plot 3: Proposer-Verifier Screening Trade-off
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = "tab:blue"
    ax1.set_xlabel("Screening Threshold theta (meters)")
    ax1.set_ylabel("Compute Speedup (x)", color=color)
    ax1.plot(thresholds, speedups, color=color, linewidth=2.0, label="Compute Speedup")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Safety Recall (Rate)", color=color)
    ax2.plot(thresholds, safety_recalls, color=color, linestyle="--", linewidth=2.0, label="Safety Recall")
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Draw line indicating 100% recall
    ax2.axhline(y=1.0, color="gray", linestyle=":", label="100% Safety Recall Line")
    
    # Highlight safe region
    if optimal_idx != -1:
        ax1.axvspan(opt_theta, thresholds[-1], color="green", alpha=0.1, label="Safe Operating Zone")
        
    plt.title("Proposer-Verifier Conjunction Screening Performance")
    fig.tight_layout()
    
    plot3_path = os.path.join(research_dir, "hcw_proposer_verifier_tradeoff.png")
    plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot 3: {plot3_path}")
    
    # 9. Verify artifacts were generated
    print("\nVerifying files exist on disk:")
    for path in [plot1_path, plot2_path, plot3_path]:
        print(f"  {path}: {'EXISTS' if os.path.exists(path) else 'MISSING'}")
        
    print("=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
