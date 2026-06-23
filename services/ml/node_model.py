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

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.05):
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

        # Final layer predicts 3D acceleration
        layers.append(nn.Linear(in_dim, 3))
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
        t_eval = t.to(state0.dtype)

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
