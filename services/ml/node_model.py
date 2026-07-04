"""Physics-Informed Neural ODE (PI-NODE) Model.

Defines the continuous-time orbital dynamics model:
    dx/dt = f_kepler(x, t) + f_neural(x, t; theta)

Where:
    - x = [rx, ry, rz, vx, vy, vz]^T is the 6D Cartesian state in TEME.
    - f_kepler is a differentiable formulation of point mass gravity and J2.
    - f_neural is an MLP that learns residual accelerations (maneuvers, drag errors).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# ── Physical Constants ──
MU_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.3781363e6  # m
J2 = 1.08262668e-3  # J2 oblateness
OMEGA_EARTH = 7.2921159e-5  # rad/s


class ResidualAccelerationNet(nn.Module):
    """Multi-Layer Perceptron to predict residual acceleration delta_a from 6D state.

    Input: [x, y, z, vx, vy, vz] (nominally normalized)
    Output: [ax_res, ay_res, az_res] (m/s²)
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.05, out_dim: int = 3):
        super().__init__()
        layers = []
        # Input features: 6 (position and velocity)
        in_dim = 6
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final layer predicts residual acceleration features
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Initialize final weights to zero so f_neural starts exactly silent
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralODEVectorField(nn.Module):
    """The continuous-time vector field function (RHS of our ODE).

    Defines:
        d_state/dt = [vx, vy, vz, ax, ay, az]
    Where:
        a = a_kepler + a_neural
    """

    def __init__(
        self,
        neural_net: nn.Module,
        bstar: float = 0.0,
        use_gravity: bool = True,
        use_j2: bool = True,
        use_drag: bool = True,
    ):
        super().__init__()
        self.neural_net = neural_net
        self.bstar = bstar
        self.use_gravity = use_gravity
        self.use_j2 = use_j2
        self.use_drag = use_drag

        # Register physical constants as buffers so they move to the correct device
        self.register_buffer("mu", torch.tensor(MU_EARTH, dtype=torch.float64))
        self.register_buffer("r_earth", torch.tensor(R_EARTH, dtype=torch.float64))
        self.register_buffer("j2", torch.tensor(J2, dtype=torch.float64))
        self.register_buffer("omega_earth", torch.tensor(OMEGA_EARTH, dtype=torch.float64))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Computes the state derivative at time t.

        Args:
            t: Scalar or 1-element tensor (time from epoch in seconds)
            state: Tensor of shape (batch_size, 6) -> [x, y, z, vx, vy, vz]

        Returns:
            derivative: Tensor of shape (batch_size, 6) -> [vx, vy, vz, ax, ay, az]
        """
        pos = state[:, :3]
        vel = state[:, 3:]
        r = torch.norm(pos, dim=-1, keepdim=True)

        # Safeguard division by zero or negative altitudes (e.g. inside Earth)
        r_safe = torch.clamp(r, min=self.r_earth / 2.0)

        # ── 1. Keplerian point-mass gravity ──
        if self.use_gravity:
            a_grav = -self.mu / (r_safe**3) * pos
        else:
            a_grav = torch.zeros_like(pos)

        # ── 2. J2 oblateness perturbation ──
        if self.use_j2 and self.use_gravity:
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            z = pos[:, 2:3]
            r2 = r_safe * r_safe
            z2 = z * z
            z_r2 = z2 / r2

            f2 = 1.5 * self.j2 * self.mu * (self.r_earth**2) / (r_safe**5)
            a_j2 = torch.cat(
                [
                    f2 * x * (5.0 * z_r2 - 1.0),
                    f2 * y * (5.0 * z_r2 - 1.0),
                    f2 * z * (5.0 * z_r2 - 3.0),
                ],
                dim=-1,
            )
        else:
            a_j2 = torch.zeros_like(pos)

        # ── 3. Atmospheric drag (exponential approximation for LEO) ──
        if self.use_drag and abs(self.bstar) > 1e-12:
            alt_km = (r.squeeze(-1) - self.r_earth) / 1000.0
            # Piecewise-like smooth exponential approximation centered around 400-500km LEO
            # rho = rho0 * exp(-(alt - h0)/H)
            rho = 4.0e-12 * torch.exp(-(alt_km - 400.0) / 55.0)
            rho = torch.clamp(rho, min=0.0).unsqueeze(-1)  # (batch, 1)

            # Velocity relative to rotating atmosphere: v_rel = v - [-omega*y, omega*x, 0]
            v_rel = vel.clone()
            v_rel[:, 0] = vel[:, 0] + self.omega_earth * pos[:, 1]
            v_rel[:, 1] = vel[:, 1] - self.omega_earth * pos[:, 0]
            v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)

            cd_a_over_m = self.bstar * 420.0  # empirical calibration factor
            a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel
        else:
            a_drag = torch.zeros_like(pos)

        a_kepler = a_grav + a_j2 + a_drag

        # ── 4. Neural residual acceleration ──
        # Normalize inputs for the MLP stability (ECI coordinates in meters are ~1e6, v ~1e3)
        pos_norm = pos / self.r_earth
        vel_norm = vel / 7500.0
        state_norm = torch.cat([pos_norm, vel_norm], dim=-1)

        a_neural = self.neural_net(state_norm)

        # ── 5. Assemble total derivatives ──
        d_pos = vel
        d_vel = a_kepler + a_neural

        return torch.cat([d_pos, d_vel], dim=-1)


class NeuralODEPropagator(nn.Module):
    """Wrapper that runs the numerical ODE solver over the vector field.

    Integrates:
        y_t = y_0 + \int_{0}^{t} f(y, tau) d_tau
    """

    def __init__(
        self,
        neural_net: nn.Module,
        bstar: float = 0.0,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        method: str = "rk4",
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.vector_field = NeuralODEVectorField(neural_net, bstar=bstar)
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.use_adjoint = use_adjoint

    def forward(self, state0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Propagate state0 forward to all timesteps in t.

        Args:
            state0: Initial Cartesian state of shape (batch_size, 6)
            t: 1D tensor of evaluation time steps (seconds from epoch), must be sorted.
               e.g. torch.tensor([0.0, 3600.0, 7200.0])

        Returns:
            trajectory: Tensor of shape (batch_size, len(t), 6)
        """
        # torchdiffeq expects t of shape (len(t),)
        # and returns shape (len(t), batch_size, 6)
        solver = odeint_adjoint if self.use_adjoint else odeint

        # Run integration
        # Note: torchdiffeq requires t to be float32 or float64 matching y0
        t_eval = t.to(device=state0.device, dtype=state0.dtype)

        # Integrate
        # If method is rk4, we can specify options like step_size if needed
        options = {}
        if self.method == "rk4":
            options = {"step_size": 30.0}  # 30 seconds step size is standard for RK4 orbit propagation

        sol = solver(
            self.vector_field,
            state0,
            t_eval,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=options,
        )

        # Reshape to (batch_size, len(t), 6) to align with standard ML shapes
        sol = sol.permute(1, 0, 2)
        return sol


class DifferentiablePotentialMLP(nn.Module):
    """3-layer MLP predicting the scalar action potential phi from position coordinates.

    Calculates the exact gradient grad_r phi analytically to avoid PyTorch
    autograd overhead inside numerical integrators.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Input features: 3 (x, y, z position coordinates)
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)

        # Initializing weights so the potential starts flat (gradient = 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc3.weight, std=1e-2)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Forward pass: computes the scalar potential phi."""
        a1 = torch.tanh(self.fc1(r))
        a2 = torch.tanh(self.fc2(a1))
        phi = self.fc3(a2)
        return phi

    def grad(self, r: torch.Tensor) -> torch.Tensor:
        """Analytical gradient d_phi/dr of shape (batch, 3)."""
        # Forward pass storing intermediate activations
        z1 = self.fc1(r)
        a1 = torch.tanh(z1)
        z2 = self.fc2(a1)
        a2 = torch.tanh(z2)

        # Backward propagation of derivatives through tanh: d/dx tanh(x) = 1 - tanh^2(x)
        d3 = self.fc3.weight  # shape (1, hidden_dim)
        d2 = d3 * (1.0 - a2 * a2)  # shape (batch, hidden_dim)
        d1 = torch.matmul(d2, self.fc2.weight) * (1.0 - a1 * a1)  # shape (batch, hidden_dim)
        g = torch.matmul(d1, self.fc1.weight)  # shape (batch, 3)
        return g


class Position3DNet(nn.Module):
    """Simple 3-layer MLP for predicting 3D fields from 3D position."""

    def __init__(self, hidden_dim: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize final weights/bias to zero so it starts silent
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class GeometricResidualODE(nn.Module):
    """The continuous-time vector field function (RHS of our ODE) with Helmholtz separation.

    Defines:
        d_state/dt = [vx, vy, vz, ax, ay, az]
    Where:
        a = a_kepler + a_conservative + a_dissipative
        a_conservative = -grad_r phi(r)
        a_dissipative = f_psi(r, v) = MLP_drag(r, v) + q * (E(r) + v x B(r))
    """

    def __init__(
        self,
        potential_net: nn.Module,
        drag_net: nn.Module,
        e_net: nn.Module,
        b_net: nn.Module,
        q_init: float = 1e-3,
        bstar: float = 0.0,
        use_gravity: bool = True,
        use_j2: bool = True,
        use_drag: bool = True,
    ):
        super().__init__()
        self.potential_net = potential_net
        self.drag_net = drag_net
        self.e_net = e_net
        self.b_net = b_net
        # Learnable charge-to-mass parameter q
        self.q = nn.Parameter(torch.tensor(q_init, dtype=torch.float64))

        self.bstar = bstar
        self.use_gravity = use_gravity
        self.use_j2 = use_j2
        self.use_drag = use_drag

        # Register physical constants as buffers so they move to the correct device
        self.register_buffer("mu", torch.tensor(MU_EARTH, dtype=torch.float64))
        self.register_buffer("r_earth", torch.tensor(R_EARTH, dtype=torch.float64))
        self.register_buffer("j2", torch.tensor(J2, dtype=torch.float64))
        self.register_buffer("omega_earth", torch.tensor(OMEGA_EARTH, dtype=torch.float64))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pos = state[:, :3]
        vel = state[:, 3:]
        r = torch.norm(pos, dim=-1, keepdim=True)
        r_safe = torch.clamp(r, min=self.r_earth / 2.0)

        # ── 1. Keplerian point-mass gravity ──
        if self.use_gravity:
            a_grav = -self.mu / (r_safe**3) * pos
        else:
            a_grav = torch.zeros_like(pos)

        # ── 2. J2 oblateness perturbation ──
        if self.use_j2 and self.use_gravity:
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            z = pos[:, 2:3]
            r2 = r_safe * r_safe
            z2 = z * z
            z_r2 = z2 / r2

            f2 = 1.5 * self.j2 * self.mu * (self.r_earth**2) / (r_safe**5)
            a_j2 = torch.cat(
                [
                    f2 * x * (5.0 * z_r2 - 1.0),
                    f2 * y * (5.0 * z_r2 - 1.0),
                    f2 * z * (5.0 * z_r2 - 3.0),
                ],
                dim=-1,
            )
        else:
            a_j2 = torch.zeros_like(pos)

        # ── 3. Atmospheric drag (exponential approximation for LEO) ──
        if self.use_drag and abs(self.bstar) > 1e-12:
            alt_km = (r.squeeze(-1) - self.r_earth) / 1000.0
            rho = 4.0e-12 * torch.exp(-(alt_km - 400.0) / 55.0)
            rho = torch.clamp(rho, min=0.0).unsqueeze(-1)  # (batch, 1)

            # Velocity relative to rotating atmosphere
            v_rel = vel.clone()
            v_rel[:, 0] = vel[:, 0] + self.omega_earth * pos[:, 1]
            v_rel[:, 1] = vel[:, 1] - self.omega_earth * pos[:, 0]
            v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)

            cd_a_over_m = self.bstar * 420.0  # empirical calibration factor
            a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel
        else:
            a_drag = torch.zeros_like(pos)

        a_kepler = a_grav + a_j2 + a_drag

        # ── 4. Neural residual acceleration ──
        # Normalization
        pos_norm = pos / self.r_earth
        vel_norm = vel / 7500.0

        # A. Conservative neural force (-grad_r phi)
        if hasattr(self.potential_net, "grad"):
            a_conservative = -self.potential_net.grad(pos_norm) * 1e-3
        else:
            with torch.enable_grad():
                p_in = pos_norm.clone().requires_grad_(True)
                phi = self.potential_net(p_in)
                grad_phi = torch.autograd.grad(
                    phi, p_in, grad_outputs=torch.ones_like(phi), create_graph=True
                )[0]
            a_conservative = -grad_phi * 1e-3

        # B. Dissipative neural force
        state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
        drag_output = self.drag_net(state_norm)

        # Check if the drag net is 1D (constrained relative velocity direction) or 3D (unconstrained)
        if drag_output.shape[-1] == 1:
            # Velocity relative to rotating atmosphere: v_rel = v - [-omega*y, omega*x, 0]
            v_rel = vel.clone()
            v_rel[:, 0] = vel[:, 0] + self.omega_earth * pos[:, 1]
            v_rel[:, 1] = vel[:, 1] - self.omega_earth * pos[:, 0]
            v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)
            v_rel_dir = v_rel / torch.clamp(v_rel_mag, min=1e-9)
            # Drag residual magnitude is scaled by 1e-5 to match physical drag acceleration scale (~10^-5 m/s^2)
            a_drag_residual = -drag_output * v_rel_dir * 1e-5
        else:
            a_drag_residual = drag_output

        # Lorentz-like force: E(r) + v_norm x B(r)
        E = self.e_net(pos_norm)
        B = self.b_net(pos_norm)
        v_cross_B = torch.cross(vel_norm, B, dim=-1)
        a_lorentz = self.q * (E + v_cross_B)

        a_dissipative = a_drag_residual + a_lorentz

        a_neural = a_conservative + a_dissipative

        # ── 5. Assemble total derivatives ──
        d_pos = vel
        d_vel = a_kepler + a_neural

        return torch.cat([d_pos, d_vel], dim=-1)


class GeometricResidualPropagator(nn.Module):
    """Wrapper that runs the numerical ODE solver over the GeometricResidualODE vector field."""

    def __init__(
        self,
        vector_field: GeometricResidualODE,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        method: str = "rk4",
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.vector_field = vector_field
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.use_adjoint = use_adjoint

    def forward(self, state0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        solver = odeint_adjoint if self.use_adjoint else odeint
        t_eval = t.to(device=state0.device, dtype=state0.dtype)

        options = {}
        if self.method == "rk4":
            options = {"step_size": 30.0}

        sol = solver(
            self.vector_field,
            state0,
            t_eval,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=options,
        )

        sol = sol.permute(1, 0, 2)
        return sol


class TrajectoryTubeSampler:
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

            # Compute Jacobian A = df/dx at x_k
            A = torch.zeros(B, 6, 6, dtype=dtype, device=device)
            for i in range(6):
                grad_outputs = torch.zeros(B, 6, dtype=dtype, device=device)
                grad_outputs[:, i] = 1.0
                x_in = x_k.clone().detach().requires_grad_(True)
                f_val = self.vector_field(t, x_in)
                grads = torch.autograd.grad(
                    f_val, x_in, grad_outputs=grad_outputs, create_graph=False, retain_graph=False
                )[0]
                A[:, i, :] = grads

            # STM: Phi = I + A * dt
            Phi = torch.eye(6, dtype=dtype, device=device).unsqueeze(0) + A * dt

            # Propagate P_next = Phi * P * Phi^T + Q * dt
            P = Phi @ P @ Phi.transpose(-1, -2) + Q * dt
            P_seq[:, k + 1, :, :] = P

        return P_seq

    def propagate_covariance_ukf(
        self, states: torch.Tensor, t_eval: torch.Tensor, P0: torch.Tensor
    ) -> torch.Tensor:
        """Propagates the initial covariance P0 along the trajectory 'states' using UKF.

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

        # UKF parameters
        L = 6
        alpha = 1e-1
        beta = 2.0
        kappa = 0.0
        lam = alpha**2 * (L + kappa) - L

        w_m = torch.zeros(13, dtype=dtype, device=device)
        w_c = torch.zeros(13, dtype=dtype, device=device)
        w_m[0] = lam / (L + lam)
        w_c[0] = lam / (L + lam) + (1.0 - alpha**2 + beta)
        w_m[1:] = 1.0 / (2.0 * (L + lam))
        w_c[1:] = 1.0 / (2.0 * (L + lam))

        P = P0.clone()
        mean_state = states[:, 0, :]

        # Setup temporary one-step ODE propagator for sigma points
        for k in range(N_steps - 1):
            t = t_eval[k]
            dt = t_eval[k + 1] - t_eval[k]

            # 1. Generate sigma points from mean_state and P
            P_spd = 0.5 * (P + P.transpose(-1, -2)) + torch.eye(6, dtype=dtype, device=device) * 1e-12
            L_chol = torch.linalg.cholesky(P_spd)
            scale = torch.sqrt(torch.tensor(L + lam, dtype=dtype, device=device))
            scaled_L_t = (scale * L_chol).transpose(-1, -2)  # (B, 6, 6)

            sigma = torch.zeros(B, 13, 6, dtype=dtype, device=device)
            sigma[:, 0, :] = mean_state
            sigma[:, 1:7, :] = mean_state.unsqueeze(1) + scaled_L_t
            sigma[:, 7:13, :] = mean_state.unsqueeze(1) - scaled_L_t

            # 2. Propagate each sigma point by dt using RK4
            sigma_flat = sigma.view(-1, 6)

            # RK4 step closure
            def rk4_step(y, t_val, dt_val, f_vf):
                k1 = f_vf(t_val, y)
                k2 = f_vf(t_val + 0.5 * dt_val, y + 0.5 * dt_val * k1)
                k3 = f_vf(t_val + 0.5 * dt_val, y + 0.5 * dt_val * k2)
                k4 = f_vf(t_val + dt_val, y + dt_val * k3)
                return y + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            sigma_prop_flat = rk4_step(sigma_flat, t, dt, self.vector_field)
            sigma_prop = sigma_prop_flat.view(B, 13, 6)

            # 3. Reconstruct mean and covariance
            mean_state = torch.sum(w_m.view(1, 13, 1) * sigma_prop, dim=1)
            diff = sigma_prop - mean_state.unsqueeze(1)
            outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
            P = torch.sum(w_c.view(1, 13, 1, 1) * outer, dim=1) + Q * dt

            P_seq[:, k + 1, :, :] = P

        return P_seq

    def sample_tube(
        self,
        states: torch.Tensor,
        P_seq: torch.Tensor,
        num_samples_per_step: int = 5,
        sigma_limit: float = 3.0,
        pos_inflation: float = 0.0,
        vel_inflation: float = 0.0,
    ) -> torch.Tensor:
        """Samples collocation points within the covariance ellipsoids along the trajectory.

        Args:
            states: Nominal trajectory states of shape (B, N_steps, 6)
            P_seq: Covariance matrices of shape (B, N_steps, 6, 6)
            num_samples_per_step: Number of off-trajectory points to sample at each time step
            sigma_limit: Mahalanobis distance limit for sampling (e.g. 3.0 for 3-sigma)
            pos_inflation: Minimum position standard deviation floor (meters)
            vel_inflation: Minimum velocity standard deviation floor (m/s)

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


class CalibratedEnsemblePropagator(nn.Module):
    """Calibrated uncertainty propagator that runs ensemble rollouts with process noise.

    Uses initial covariance inflation and integrated process noise spectral density
    to match the 95% confidence ellipsoid for orbital prediction tasks.
    """

    def __init__(
        self,
        propagator: nn.Module,
        ensemble_size: int = 50,
        inflation_factor: float = 50.0,
        q_acc: float = 1.0e-3,
    ):
        super().__init__()
        self.propagator = propagator
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor
        self.q_acc = q_acc

    def forward(
        self, state0: torch.Tensor, cov0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagates mean and uncertainty forward.

        Args:
            state0: Nominal initial state of shape (batch_size, 6) or (6,)
            cov0: Initial state covariance of shape (batch_size, 6, 6) or (6, 6)
            t: 1D tensor of evaluation time steps (seconds from epoch)

        Returns:
            mean_trajectory: Tensor of shape (batch_size, len(t), 6)
            ensemble_trajectories: Tensor of shape (batch_size, ensemble_size, len(t), 6)
            calibrated_covariances: Tensor of shape (batch_size, len(t), 6, 6)
        """
        # Ensure correct batch dimensions
        if state0.ndim == 1:
            state0 = state0.unsqueeze(0)  # (1, 6)
        if cov0.ndim == 2:
            cov0 = cov0.unsqueeze(0)  # (1, 6, 6)

        B, _ = state0.shape
        N_steps = len(t)
        device = state0.device
        dtype = state0.dtype

        # 1. Generate initial ensemble using Cholesky factorization of cov0
        # cov0_spd is enforced to be symmetric positive definite
        cov0_spd = 0.5 * (cov0 + cov0.transpose(-1, -2)) + torch.eye(6, dtype=dtype, device=device) * 1e-12
        L = torch.linalg.cholesky(cov0_spd)  # (B, 6, 6)

        # Sample ensemble member initial states: shape (B, M, 6)
        ensemble0 = torch.zeros(B, self.ensemble_size, 6, dtype=dtype, device=device)
        for b in range(B):
            # Sample standard normal vector in 6D
            z = torch.randn(self.ensemble_size, 6, dtype=dtype, device=device)  # (M, 6)
            # Project onto the Cholesky factor
            perturb = torch.matmul(z, L[b].T)  # (M, 6)
            ensemble0[b] = state0[b].unsqueeze(0) + perturb

        # 2. Run parallel deterministic propagations for all ensemble members
        # Flatten batch and ensemble dimensions for batch propagation: shape (B * M, 6)
        ensemble0_flat = ensemble0.view(B * self.ensemble_size, 6)
        
        # Propagate flat ensemble: returns shape (B * M, N_steps, 6)
        sol_flat = self.propagator(ensemble0_flat, t)
        
        # Reshape to (B, M, N_steps, 6)
        sol_ens = sol_flat.view(B, self.ensemble_size, N_steps, 6)

        # Also propagate nominal trajectory
        sol_nom = self.propagator(state0, t)  # (B, N_steps, 6)

        # 3. Calculate calibrated covariances at each timestep
        calibrated_covs = torch.zeros(B, N_steps, 6, 6, dtype=dtype, device=device)
        for k in range(N_steps):
            # Time from epoch in seconds
            t_sec = t[k].item()
            
            # Position-position process noise growth: q_acc * (t^3 / 3) * eye(3)
            # Velocity-velocity process noise growth: q_acc * t * eye(3)
            # Position-velocity cross noise growth: q_acc * (t^2 / 2) * eye(3)
            q_term_pos = self.q_acc * (t_sec**3 / 3.0)
            q_term_vel = self.q_acc * t_sec
            q_term_cross = self.q_acc * (t_sec**2 / 2.0)
            
            # Build Q_d block covariance
            Q_d = torch.zeros(6, 6, dtype=dtype, device=device)
            Q_d[0:3, 0:3] = torch.eye(3, dtype=dtype, device=device) * q_term_pos
            Q_d[3:6, 3:6] = torch.eye(3, dtype=dtype, device=device) * q_term_vel
            Q_d[0:3, 3:6] = torch.eye(3, dtype=dtype, device=device) * q_term_cross
            Q_d[3:6, 0:3] = torch.eye(3, dtype=dtype, device=device) * q_term_cross

            for b in range(B):
                # Retrieve ensemble states at step k: shape (M, 6)
                ens_k = sol_ens[b, :, k, :]
                
                # Compute sample covariance: shape (6, 6)
                mean_k = torch.mean(ens_k, dim=0, keepdim=True)
                diff_k = ens_k - mean_k
                sample_cov = torch.matmul(diff_k.T, diff_k) / (self.ensemble_size - 1)
                
                # Apply initial covariance inflation factor
                cov_calibrated = sample_cov * self.inflation_factor
                
                # Add integrated process noise spectral density
                cov_calibrated += Q_d
                
                # Ensure symmetric positive definite regularization
                cov_calibrated = 0.5 * (cov_calibrated + cov_calibrated.T) + torch.eye(6, dtype=dtype, device=device) * 1e-6
                calibrated_covs[b, k, :, :] = cov_calibrated

        # Return nominal trajectory, ensemble, and sequence of calibrated covariances
        return sol_nom, sol_ens, calibrated_covs
