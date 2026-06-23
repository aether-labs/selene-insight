"""Scalar Potential Neural ODE model (PIML-fluid).

Defines the continuous-time orbital dynamics model:
    dx/dt = f_kepler(x, t) + f_neural(x, t; theta)

Where the neural residual is conservative and derived from a scalar potential:
    f_neural(x, t; theta) = -grad_r phi(r, t; theta)

Using analytical backpropagation for exact gradient computation without
autograd tracing overhead inside numerical integrators.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# ── Physical Constants ──
MU_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.3781363e6  # m
J2 = 1.08262668e-3
J3 = -2.53265648e-6
J4 = -1.61962159e-6
OMEGA_EARTH = 7.2921159e-5


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
        nn.init.zeros_(self.fc3.weight)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Forward pass: computes the scalar potential phi.

        Args:
            r: Tensor of shape (batch, 3) representing coordinates

        Returns:
            phi: Tensor of shape (batch, 1)
        """
        a1 = torch.tanh(self.fc1(r))
        a2 = torch.tanh(self.fc2(a1))
        phi = self.fc3(a2)
        return phi

    def grad(self, r: torch.Tensor) -> torch.Tensor:
        """Analytical gradient d_phi/dr of shape (batch, 3).

        Args:
            r: Tensor of shape (batch, 3) representing coordinates

        Returns:
            grad_r_phi: Tensor of shape (batch, 3)
        """
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


class ScalarPotentialNeuralODEVectorField(nn.Module):
    """RHS of the ODE with J2+J3+J4+drag physics and scalar potential neural forces.

    dx/dt = [vx, vy, vz, ax, ay, az]
    Where:
        ax, ay, az = a_kepler + a_drag - grad_r phi
    """

    def __init__(
        self,
        potential_net: DifferentiablePotentialMLP,
        bstar: float = 0.0,
        use_gravity: bool = True,
        use_j2: bool = True,
        use_j3: bool = True,
        use_j4: bool = True,
        use_drag: bool = True,
    ):
        super().__init__()
        self.potential_net = potential_net
        self.bstar = bstar
        self.use_gravity = use_gravity
        self.use_j2 = use_j2
        self.use_j3 = use_j3
        self.use_j4 = use_j4
        self.use_drag = use_drag

        # Register physical constants as double precision buffers
        self.register_buffer("mu", torch.tensor(MU_EARTH, dtype=torch.float64))
        self.register_buffer("r_earth", torch.tensor(R_EARTH, dtype=torch.float64))
        self.register_buffer("j2", torch.tensor(J2, dtype=torch.float64))
        self.register_buffer("j3", torch.tensor(J3, dtype=torch.float64))
        self.register_buffer("j4", torch.tensor(J4, dtype=torch.float64))
        self.register_buffer("omega_earth", torch.tensor(OMEGA_EARTH, dtype=torch.float64))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pos = state[:, :3]
        vel = state[:, 3:]
        r = torch.norm(pos, dim=-1, keepdim=True)

        # Safeguard division by zero
        r_safe = torch.clamp(r, min=self.r_earth / 2.0)

        # ── 1. Point-mass gravity ──
        if self.use_gravity:
            a_grav = -self.mu / (r_safe**3) * pos
        else:
            a_grav = torch.zeros_like(pos)

        # ── 2. Zonal Gravity Harmonics (J2 + J3 + J4) ──
        a_harmonics = torch.zeros_like(pos)
        if self.use_gravity:
            x = pos[:, 0:1]
            y = pos[:, 1:2]
            z = pos[:, 2:3]
            r2 = r_safe * r_safe
            z2 = z * z
            z_r2 = z2 / r2

            # J2 oblateness
            if self.use_j2:
                f2 = 1.5 * self.j2 * self.mu * (self.r_earth**2) / (r_safe**5)
                a_j2 = torch.cat(
                    [
                        f2 * x * (5.0 * z_r2 - 1.0),
                        f2 * y * (5.0 * z_r2 - 1.0),
                        f2 * z * (5.0 * z_r2 - 3.0),
                    ],
                    dim=-1,
                )
                a_harmonics += a_j2

            # J3 pear-shape perturbation
            if self.use_j3:
                f3 = 0.5 * self.j3 * self.mu * (self.r_earth**3) / (r_safe**7)
                a_j3 = torch.cat(
                    [
                        f3 * 5.0 * x * (7.0 * z * z_r2 - 3.0 * z),
                        f3 * 5.0 * y * (7.0 * z * z_r2 - 3.0 * z),
                        f3 * (6.0 * z2 - 7.0 * z2 * z_r2 - 0.6 * r2),
                    ],
                    dim=-1,
                )
                a_harmonics += a_j3

            # J4 perturbation
            if self.use_j4:
                z_r4 = z_r2 * z_r2
                f4 = -0.625 * self.j4 * self.mu * (self.r_earth**4) / (r_safe**7)
                a_j4 = torch.cat(
                    [
                        f4 * x / r2 * (3.0 - 42.0 * z_r2 + 63.0 * z_r4),
                        f4 * y / r2 * (3.0 - 42.0 * z_r2 + 63.0 * z_r4),
                        f4 * z / r2 * (15.0 - 70.0 * z_r2 + 63.0 * z_r4),
                    ],
                    dim=-1,
                )
                a_harmonics += a_j4

        # ── 3. Atmospheric drag ──
        if self.use_drag and abs(self.bstar) > 1e-12:
            alt_km = (r.squeeze(-1) - self.r_earth) / 1000.0
            rho = 4.0e-12 * torch.exp(-(alt_km - 400.0) / 55.0)
            rho = torch.clamp(rho, min=0.0).unsqueeze(-1)

            # Relative velocity
            v_rel = vel.clone()
            v_rel[:, 0] = vel[:, 0] + self.omega_earth * pos[:, 1]
            v_rel[:, 1] = vel[:, 1] - self.omega_earth * pos[:, 0]
            v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)

            cd_a_over_m = self.bstar * 420.0
            a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel
        else:
            a_drag = torch.zeros_like(pos)

        a_kepler = a_grav + a_harmonics + a_drag

        # ── 4. Scalar Potential Neural Acceleration ──
        # Input to potential network is normalized position
        pos_norm = pos / self.r_earth
        # Acceleration residual is -grad_r phi
        # Chain rule: d_phi / d_r = (d_phi / d_r_norm) * (d_r_norm / d_r) = (d_phi / d_r_norm) / R_EARTH
        a_neural = -self.potential_net.grad(pos_norm) / self.r_earth

        # Total derivatives
        d_pos = vel
        d_vel = a_kepler + a_neural

        return torch.cat([d_pos, d_vel], dim=-1)


class ScalarPotentialNeuralODEPropagator(nn.Module):
    """Wrapper that runs the ODE solver over the scalar potential vector field."""

    def __init__(
        self,
        potential_net: DifferentiablePotentialMLP,
        bstar: float = 0.0,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        method: str = "rk4",
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.vector_field = ScalarPotentialNeuralODEVectorField(potential_net, bstar=bstar)
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.use_adjoint = use_adjoint

    def forward(self, state0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        solver = odeint_adjoint if self.use_adjoint else odeint
        t_eval = t.to(state0.dtype)

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
