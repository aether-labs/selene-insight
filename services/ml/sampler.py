"""Trajectory Tube Sampler class for generating training collocation points by sampling within covariance ellipsoids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class TrajectoryTubeSampler(nn.Module):
    """Generates collocation points by propagating covariance ellipsoids and sampling within them.

    Supports propagation via:
      - STM (State Transition Matrix / Extended Kalman Filter style)
      - UKF (Unscented Kalman Filter / Unscented Transform style)
    """

    def __init__(self, vector_field: nn.Module, q_process_noise: float = 1e-10):
        super().__init__()
        self.vector_field = vector_field
        # Process noise diagonal coefficient
        self.q_coef = q_process_noise

    def propagate_covariance_stm(
        self, states: torch.Tensor, t_eval: torch.Tensor, P0: torch.Tensor
    ) -> torch.Tensor:
        """Propagates the initial covariance P0 along the trajectory 'states' using STM.

        Args:
            states: Tensor of shape (B, N_steps, 6) - nominal trajectory states
            t_eval: Tensor of shape (N_steps,) - time steps
            P0: Tensor of shape (B, 6, 6) - initial covariance

        Returns:
            P_seq: Tensor of shape (B, N_steps, 6, 6) - propagated covariance matrices
        """
        B, N_steps, _ = states.shape
        device = states.device
        dtype = states.dtype

        P_seq = torch.zeros(B, N_steps, 6, 6, dtype=dtype, device=device)
        P_seq[:, 0, :, :] = P0

        # Process noise covariance matrix
        Q = torch.eye(6, dtype=dtype, device=device) * self.q_coef

        P = P0.clone()
        for k in range(N_steps - 1):
            t = t_eval[k]
            dt = t_eval[k + 1] - t_eval[k]
            x_k = states[:, k, :]

            # Compute Jacobian A(t, x_k) of shape (B, 6, 6)
            # Since df/dx needs autograd, we evaluate df/dx per batch element
            A = torch.zeros(B, 6, 6, dtype=dtype, device=device)
            x_in = x_k.clone().detach().requires_grad_(True)
            f_val = self.vector_field(t, x_in)

            for i in range(6):
                grad_outputs = torch.zeros(B, 6, dtype=dtype, device=device)
                grad_outputs[:, i] = 1.0
                grads = torch.autograd.grad(
                    f_val, x_in, grad_outputs=grad_outputs, retain_graph=True
                )[0]
                A[:, i, :] = grads

            # STM approximation: Phi = I + A * dt
            Phi = torch.eye(6, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1) + A * dt

            # Propagate: P_{k+1} = Phi * P_k * Phi^T + Q * dt
            P = torch.matmul(torch.matmul(Phi, P), Phi.transpose(-1, -2)) + Q * dt
            P_seq[:, k + 1, :, :] = P

        return P_seq

    def propagate_covariance_ukf(
        self, states: torch.Tensor, t_eval: torch.Tensor, P0: torch.Tensor
    ) -> torch.Tensor:
        """Propagates the initial covariance P0 along the trajectory using UKF Unscented Transform.

        Args:
            states: Tensor of shape (B, N_steps, 6) - nominal trajectory states
            t_eval: Tensor of shape (N_steps,) - time steps
            P0: Tensor of shape (B, 6, 6) - initial covariance

        Returns:
            P_seq: Tensor of shape (B, N_steps, 6, 6) - propagated covariance matrices
        """
        B, N_steps, _ = states.shape
        device = states.device
        dtype = states.dtype

        P_seq = torch.zeros(B, N_steps, 6, 6, dtype=dtype, device=device)
        P_seq[:, 0, :, :] = P0

        # UKF Parameters: L=6
        L_dim = 6
        alpha = 1e-1
        beta = 2.0
        kappa = 0.0

        lambd = (alpha**2) * (L_dim + kappa) - L_dim
        c = L_dim + lambd

        # Weights
        w_m = torch.zeros(2 * L_dim + 1, dtype=dtype, device=device)
        w_c = torch.zeros(2 * L_dim + 1, dtype=dtype, device=device)
        w_m[0] = lambd / c
        w_c[0] = (lambd / c) + (1.0 - alpha**2 + beta)
        for i in range(1, 2 * L_dim + 1):
            w_m[i] = 1.0 / (2.0 * c)
            w_c[i] = 1.0 / (2.0 * c)

        # Process noise
        Q = torch.eye(6, dtype=dtype, device=device) * self.q_coef

        P = P0.clone()
        for k in range(N_steps - 1):
            t = t_eval[k]
            dt = t_eval[k + 1] - t_eval[k]
            x_k = states[:, k, :]

            # For each batch element, compute sigma points
            sigma_pts = torch.zeros(B, 2 * L_dim + 1, 6, dtype=dtype, device=device)
            sigma_pts[:, 0, :] = x_k

            # Regularize P to be strictly SPD
            P_spd = 0.5 * (P + P.transpose(-1, -2)) + torch.eye(6, dtype=dtype, device=device) * 1e-12
            L_mat = torch.linalg.cholesky(P_spd)  # (B, 6, 6)
            sqrt_c = np.sqrt(c)

            for i in range(L_dim):
                sigma_pts[:, i + 1, :] = x_k + sqrt_c * L_mat[:, :, i]
                sigma_pts[:, L_dim + i + 1, :] = x_k - sqrt_c * L_mat[:, :, i]

            # Propagate all sigma points through the vector field
            # We flatten to shape (B * (2L+1), 6)
            sigma_pts_flat = sigma_pts.view(-1, 6)
            # Evaluate using RK4 integration over dt
            k1 = self.vector_field(t, sigma_pts_flat)
            k2 = self.vector_field(t + 0.5 * dt, sigma_pts_flat + 0.5 * dt * k1)
            k3 = self.vector_field(t + 0.5 * dt, sigma_pts_flat + 0.5 * dt * k2)
            k4 = self.vector_field(t + dt, sigma_pts_flat + dt * k3)
            sigma_pts_prop_flat = sigma_pts_flat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Reshape back: (B, 2L+1, 6)
            sigma_pts_prop = sigma_pts_prop_flat.view(B, 2 * L_dim + 1, 6)

            # Reconstruct mean and covariance
            mean_prop = torch.sum(w_m.view(1, -1, 1) * sigma_pts_prop, dim=1)  # (B, 6)

            P_new = torch.zeros(B, 6, 6, dtype=dtype, device=device)
            for i in range(2 * L_dim + 1):
                diff = sigma_pts_prop[:, i, :] - mean_prop
                P_new += w_c[i] * torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2))

            P = P_new + Q * dt
            P_seq[:, k + 1, :, :] = P

        return P_seq

    def generate_collocation_points(
        self,
        states: torch.Tensor,
        P_seq: torch.Tensor,
        num_samples_per_step: int = 10,
        sigma_limit: float = 3.0,
        pos_inflation: float = 10.0,
        vel_inflation: float = 0.05,
    ) -> torch.Tensor:
        """Generates off-trajectory collocation points within the uncertainty tube.

        Args:
            states: Tensor of shape (B, N_steps, 6) - nominal trajectory states
            P_seq: Tensor of shape (B, N_steps, 6, 6) - sequence of covariances
            num_samples_per_step: int - samples to draw per time step
            sigma_limit: float - maximum scaling boundary of the ellipsoid
            pos_inflation: float - constant isotropic position variance offset
            vel_inflation: float - constant isotropic velocity variance offset

        Returns:
            sampled_states: Tensor of shape (B, N_steps, num_samples_per_step, 6)
        """
        B, N_steps, _ = states.shape
        device = states.device
        dtype = states.dtype

        sampled_states = torch.zeros(B, N_steps, num_samples_per_step, 6, dtype=dtype, device=device)

        # Create diagonal inflation matrix
        inflation_diag = torch.tensor(
            [pos_inflation**2] * 3 + [vel_inflation**2] * 3,
            dtype=dtype,
            device=device,
        )
        inflation_matrix = torch.diag(inflation_diag).unsqueeze(0)  # (1, 6, 6)

        for k in range(N_steps):
            P = P_seq[:, k, :, :]  # (B, 6, 6)
            mean = states[:, k, :]  # (B, 6)

            # Apply isotropic diagonal inflation
            P_inflated = P + inflation_matrix

            # Ensure P is SPD
            P_spd = 0.5 * (P_inflated + P_inflated.transpose(-1, -2)) + torch.eye(6, dtype=dtype, device=device) * 1e-12
            L_chol = torch.linalg.cholesky(P_spd)  # (B, 6, 6)

            for s in range(num_samples_per_step):
                # 1. Sample standard normal vector in 6D
                z = torch.randn(B, 6, dtype=dtype, device=device)
                z_norm = torch.norm(z, dim=-1, keepdim=True)
                z_dir = z / torch.clamp(z_norm, min=1e-9)

                # 2. Sample radius from uniform volume distribution
                u = torch.rand(B, 1, dtype=dtype, device=device)
                r = sigma_limit * (u ** (1.0 / 6.0))  # 6D radius

                # 3. Shift and scale by Cholesky factor
                scaled_perturb = L_chol @ (r * z_dir).unsqueeze(-1)  # (B, 6, 1)
                sampled_states[:, k, s, :] = mean + scaled_perturb.squeeze(-1)

        return sampled_states
