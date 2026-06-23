"""Hamiltonian Neural Network (HNN) models for continuous-time orbital dynamics.

Defines the continuous-time orbital dynamics model:
    dx/dt = f_kepler(x, t) + f_neural_hamiltonian(x, t; theta)

Where the neural residual is governed by Hamilton's equations derived from a
neural Hamiltonian scalar function H_neural(q, p; theta).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# ── Physical Constants ──
MU_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.3781363e6  # m
J2 = 1.08262668e-3
OMEGA_EARTH = 7.2921159e-5
V_NORM = 7500.0  # m/s reference orbit velocity


class DifferentiableHamiltonianMLP(nn.Module):
    """3-layer MLP predicting the scalar residual energy H_neural from state.

    Input features: 6 (normalized position q and normalized momentum p)
    Output: 1 (scalar energy value)

    Calculates analytical gradients [dH/dq, dH/dp] directly to avoid autograd
    overhead inside numerical integration loops.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)

        # Initializing weights so the energy starts flat (gradient = 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: computes the scalar neural Hamiltonian.

        Args:
            state: Tensor of shape (batch, 6) representing [q_norm, p_norm]
        """
        a1 = torch.tanh(self.fc1(state))
        a2 = torch.tanh(self.fc2(a1))
        h_val = self.fc3(a2)
        return h_val

    def grad(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytical gradient dH/dq and dH/dp of shape (batch, 3).

        Args:
            state: Tensor of shape (batch, 6) representing [q_norm, p_norm]

        Returns:
            grad_q: dH/dq_norm of shape (batch, 3)
            grad_p: dH/dp_norm of shape (batch, 3)
        """
        z1 = self.fc1(state)
        a1 = torch.tanh(z1)
        z2 = self.fc2(a1)
        a2 = torch.tanh(z2)

        # Backward propagation of derivatives through tanh
        d3 = self.fc3.weight  # shape (1, hidden_dim)
        d2 = d3 * (1.0 - a2 * a2)  # shape (batch, hidden_dim)
        d1 = torch.matmul(d2, self.fc2.weight) * (1.0 - a1 * a1)  # shape (batch, hidden_dim)
        g = torch.matmul(d1, self.fc1.weight)  # shape (batch, 6)

        # Split into grad_q and grad_p
        grad_q = g[:, :3]
        grad_p = g[:, 3:]
        return grad_q, grad_p


class HamiltonianNeuralODEVectorField(nn.Module):
    """RHS of the ODE with J2+drag physics and Hamiltonian neural forces.

    Non-separable formulation:
        dq/dt = velocity + dH_neural / dp
        dp/dt = a_physics - dH_neural / dq
    """

    def __init__(
        self,
        hamiltonian_net: DifferentiableHamiltonianMLP,
        bstar: float = 0.0,
        use_gravity: bool = True,
        use_j2: bool = True,
        use_drag: bool = True,
        separable: bool = False,
    ):
        super().__init__()
        self.hamiltonian_net = hamiltonian_net
        self.bstar = bstar
        self.use_gravity = use_gravity
        self.use_j2 = use_j2
        self.use_drag = use_drag
        self.separable = separable

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

        # ── 3. Atmospheric drag ──
        if self.use_drag and abs(self.bstar) > 1e-12:
            alt_km = (r.squeeze(-1) - self.r_earth) / 1000.0
            rho = 4.0e-12 * torch.exp(-(alt_km - 400.0) / 55.0)
            rho = torch.clamp(rho, min=0.0).unsqueeze(-1)

            v_rel = vel.clone()
            v_rel[:, 0] = vel[:, 0] + self.omega_earth * pos[:, 1]
            v_rel[:, 1] = vel[:, 1] - self.omega_earth * pos[:, 0]
            v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)

            cd_a_over_m = self.bstar * 420.0
            a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel
        else:
            a_drag = torch.zeros_like(pos)

        a_physics = a_grav + a_j2 + a_drag

        # ── 4. Hamiltonian Neural Forces ──
        # Inputs normalized for numerical conditioning
        pos_norm = pos / self.r_earth
        vel_norm = vel / V_NORM
        
        if self.separable:
            # Separable HNN (Equivalent to Scalar Potential): neural force depends only on position
            # V_neural = potential(pos_norm), Hamiltonian has no p-coupling
            # dp/dt = a_physics - dV_neural/dq
            # dq/dt = velocity (no p-gradients)
            grad_q = self.hamiltonian_net.grad(torch.cat([pos_norm, torch.zeros_like(vel_norm)], dim=-1))[0]
            grad_p = torch.zeros_like(vel_norm)
        else:
            # Coupled HNN: full state input
            state_norm = torch.cat([pos_norm, vel_norm], dim=-1)
            grad_q, grad_p = self.hamiltonian_net.grad(state_norm)

        # Chain rule corrections
        # dx/dt = vel + dH/dp_norm * (1/V_NORM)
        # dv/dt = a_physics - dH/dq_norm * (1/R_EARTH)
        d_pos = vel + grad_p / V_NORM
        d_vel = a_physics - grad_q / self.r_earth

        return torch.cat([d_pos, d_vel], dim=-1)


class HamiltonianNeuralODEPropagator(nn.Module):
    """Wrapper that runs the ODE solver over the Hamiltonian vector field."""

    def __init__(
        self,
        hamiltonian_net: DifferentiableHamiltonianMLP,
        bstar: float = 0.0,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        method: str = "rk4",
        use_adjoint: bool = False,
        separable: bool = False,
    ):
        super().__init__()
        self.vector_field = HamiltonianNeuralODEVectorField(
            hamiltonian_net, bstar=bstar, separable=separable
        )
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
