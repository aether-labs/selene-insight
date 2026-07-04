"""Task T-097 verification: regularized residuals, value fields, maneuvers.

This script is a self-contained synthetic validation harness for 2D planar
HCW relative motion in LEO. It trains compact neural models for:

1. Residual-gradient regularization in a physics-anchored residual ODE.
2. Safety proposer baselines used ahead of an exact rollout verifier.
3. Conservative/dissipative residual separation with maneuver isolation.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 97
DEVICE = torch.device("cpu")
DTYPE = torch.float32

MU_EARTH = 3.986004418e14
R_EARTH = 6.3781363e6
ALTITUDE = 500_000.0
R_ORBIT = R_EARTH + ALTITUDE
MEAN_MOTION = math.sqrt(MU_EARTH / (R_ORBIT**3))

KEEP_OUT_M = 150.0
POS_SCALE = 7_500.0
VEL_SCALE = 5.0
ACCEL_SCALE = 100_000.0

# Synthetic 2D perturbations. The first term is conservative and position-only;
# the second term is non-conservative drag aligned with -v.
C_J2_CUBIC = 1.0e-24
C_DRAG = 2.0e-8

RESEARCH_DIR = Path("/Users/yong/projects/substratum-internal/memory/research")


def set_deterministic(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def normalize_state(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[:, :2] / POS_SCALE, state[:, 2:] / VEL_SCALE), dim=-1)


def base_hcw_accel(state: torch.Tensor) -> torch.Tensor:
    x = state[:, 0:1]
    vx = state[:, 2:3]
    vy = state[:, 3:4]
    ax = 3.0 * (MEAN_MOTION**2) * x + 2.0 * MEAN_MOTION * vy
    ay = -2.0 * MEAN_MOTION * vx
    return torch.cat((ax, ay), dim=-1)


def conservative_residual(state: torch.Tensor) -> torch.Tensor:
    r = state[:, :2]
    return -C_J2_CUBIC * (r**3)


def drag_residual(state: torch.Tensor) -> torch.Tensor:
    v = state[:, 2:]
    speed = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
    return -C_DRAG * speed * v


def residual_accel(state: torch.Tensor) -> torch.Tensor:
    return conservative_residual(state) + drag_residual(state)


def true_accel(state: torch.Tensor) -> torch.Tensor:
    return base_hcw_accel(state) + residual_accel(state)


def hcw_rhs(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[:, 2:], base_hcw_accel(state)), dim=-1)


def true_rhs(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[:, 2:], true_accel(state)), dim=-1)


def rk4_step(state: torch.Tensor, dt: float, rhs_fn) -> torch.Tensor:
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * dt * k1)
    k3 = rhs_fn(state + 0.5 * dt * k2)
    k4 = rhs_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(state0: torch.Tensor, t_eval: torch.Tensor, rhs_fn, maneuvers=None) -> torch.Tensor:
    states = torch.zeros(
        state0.shape[0], len(t_eval), state0.shape[1], dtype=state0.dtype, device=state0.device
    )
    states[:, 0] = state0
    state = state0.clone()
    burns_by_step = {}
    if maneuvers:
        for burn in maneuvers:
            burns_by_step.setdefault(burn.step_idx, []).append(burn.delta_v)

    for idx in range(len(t_eval) - 1):
        dt = float(t_eval[idx + 1] - t_eval[idx])
        state = rk4_step(state, dt, rhs_fn)
        for delta_v in burns_by_step.get(idx + 1, []):
            state[:, 2:] += delta_v.to(state.device, state.dtype)
        states[:, idx + 1] = state
    return states


def residual_rhs_from_net(net: nn.Module):
    def rhs(state: torch.Tensor) -> torch.Tensor:
        a_res = net(state) / ACCEL_SCALE
        return torch.cat((state[:, 2:], base_hcw_accel(state) + a_res), dim=-1)

    return rhs


class ResidualNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_state(state))


class ProposerMLP(nn.Module):
    def __init__(self, out_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_state(state))


class PotentialNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pos_norm: torch.Tensor) -> torch.Tensor:
        return self.net(pos_norm)


class DragMagnitudeNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(normalize_state(state)))


class HelmholtzResidualModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.potential = PotentialNet(hidden_dim)
        self.drag = DragMagnitudeNet(hidden_dim)

    def components_scaled(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_norm = (state[:, :2] / POS_SCALE).detach().clone().requires_grad_(True)
        phi = self.potential(pos_norm).sum()
        grad_phi = torch.autograd.grad(phi, pos_norm, create_graph=self.training)[0]
        a_cons_scaled = -grad_phi

        v = state[:, 2:]
        speed = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
        anti_v = -v / speed
        a_drag_scaled = self.drag(state) * anti_v
        return a_cons_scaled, a_drag_scaled

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a_cons_scaled, a_drag_scaled = self.components_scaled(state)
        return a_cons_scaled + a_drag_scaled


@dataclass
class Burn:
    step_idx: int
    time_s: float
    delta_v: torch.Tensor

    @property
    def magnitude(self) -> float:
        return float(torch.linalg.norm(self.delta_v).item())


def sample_initial_states(n_samples: int, seed: int, near_conjunction: bool = False) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    if near_conjunction:
        r_chunks = []
        v_chunks = []
        while sum(len(chunk) for chunk in r_chunks) < n_samples:
            batch = max(256, n_samples * 2)
            r_try = rng.normal(loc=(-2_000.0, 2_500.0), scale=(900.0, 950.0), size=(batch, 2))
            t_ca = rng.uniform(3_000.0, 12_000.0, size=batch)
            v_try = np.array([solve_hcw_intercept_velocity(r, t) for r, t in zip(r_try, t_ca, strict=True)])
            v_try += rng.normal(0.0, 0.20, size=(batch, 2))
            keep = np.linalg.norm(v_try, axis=1) < 4.5
            r_chunks.append(r_try[keep])
            v_chunks.append(v_try[keep])
        r0 = np.vstack(r_chunks)[:n_samples]
        v0 = np.vstack(v_chunks)[:n_samples]
    else:
        x0 = rng.uniform(-3_500.0, 3_500.0, size=(n_samples, 1))
        y0 = rng.uniform(-4_000.0, 4_000.0, size=(n_samples, 1))
        vx0 = rng.uniform(-0.45, 0.45, size=(n_samples, 1))
        # HCW bounded-relative-orbit condition suppresses the secular y drift.
        vy0 = -2.0 * MEAN_MOTION * x0 + rng.normal(0.0, 0.08, size=(n_samples, 1))
        r0 = np.hstack((x0, y0))
        v0 = np.hstack((vx0, vy0))
    states = np.hstack((r0, v0)).astype(np.float32)
    return torch.tensor(states, dtype=DTYPE, device=DEVICE)


def solve_hcw_intercept_velocity(r0: np.ndarray, t_s: float) -> np.ndarray:
    nt = MEAN_MOTION * t_s
    c = math.cos(nt)
    s = math.sin(nt)
    m_pos = np.array([[4.0 - 3.0 * c, 0.0], [6.0 * (s - nt), 1.0]])
    m_vel = np.array(
        [
            [s / MEAN_MOTION, 2.0 * (1.0 - c) / MEAN_MOTION],
            [-2.0 * (1.0 - c) / MEAN_MOTION, (4.0 * s - 3.0 * nt) / MEAN_MOTION],
        ]
    )
    return np.linalg.solve(m_vel, -(m_pos @ r0))


def build_collocation_dataset(n_points: int = 12_000) -> tuple[torch.Tensor, torch.Tensor]:
    n_seed_states = max(32, n_points // 160)
    s0 = sample_initial_states(n_seed_states, seed=SEED + 1, near_conjunction=False)
    t_eval = torch.arange(0.0, 72.0 * 3600.0 + 900.0, 900.0, dtype=DTYPE, device=DEVICE)
    traj_states = propagate(s0, t_eval, true_rhs).reshape(-1, 4)
    finite = torch.isfinite(traj_states).all(dim=1)
    bounded = torch.linalg.norm(traj_states[:, :2], dim=-1) < 250_000.0
    states = traj_states[finite & bounded]
    if len(states) < n_points:
        extra = sample_initial_states(n_points - len(states), seed=SEED + 11, near_conjunction=False)
        states = torch.cat((states, extra), dim=0)
    idx = torch.randperm(len(states), device=DEVICE)[:n_points]
    states = states[idx]
    targets = residual_accel(states) * ACCEL_SCALE
    noise = torch.randn_like(targets) * 0.01 * targets.std(dim=0, keepdim=True).clamp_min(1.0)
    return states, targets + noise


def jacobian_regularization(net: nn.Module, states: torch.Tensor) -> torch.Tensor:
    x = states.detach().clone().requires_grad_(True)
    pred = net(x) / ACCEL_SCALE
    reg = torch.zeros((), dtype=x.dtype, device=x.device)
    for j in range(2):
        grad_j = torch.autograd.grad(pred[:, j].sum(), x, create_graph=True)[0][:, :2]
        reg = reg + torch.mean(torch.sum(grad_j**2, dim=-1))
    return reg


def train_residual_net(
    states: torch.Tensor, targets: torch.Tensor, lambda_jac: float, epochs: int = 320
) -> tuple[ResidualNet, list[float]]:
    net = ResidualNet().to(DEVICE, DTYPE)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-5)
    losses: list[float] = []
    batch_size = 768
    n_samples = states.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            xb = states[idx]
            yb = targets[idx]
            opt.zero_grad()
            pred = net(xb)
            fit_loss = F.mse_loss(pred, yb)
            loss = fit_loss
            if lambda_jac > 0.0:
                loss = loss + lambda_jac * jacobian_regularization(net, xb)
            loss.backward()
            opt.step()
            epoch_loss += float(fit_loss.item()) * len(idx)
        losses.append(epoch_loss / n_samples)
    return net, losses


def evaluate_regularization() -> dict[str, object]:
    print("\n[Experiment 1] Residual-gradient regularization")
    states, targets = build_collocation_dataset()
    t0 = time.time()
    net_plain, loss_plain = train_residual_net(states, targets, lambda_jac=0.0)
    net_reg, loss_reg = train_residual_net(states, targets, lambda_jac=0.01)
    print(f"  Trained two residual fields in {time.time() - t0:.1f}s")
    print(f"  Final collocation MSE: lambda=0 {loss_plain[-1]:.4e}, lambda=0.01 {loss_reg[-1]:.4e}")

    t_eval = torch.arange(0.0, 72.0 * 3600.0 + 600.0, 600.0, dtype=DTYPE, device=DEVICE)
    test_s0 = sample_initial_states(32, seed=SEED + 2, near_conjunction=False)
    truth = propagate(test_s0, t_eval, true_rhs)
    with torch.no_grad():
        base = propagate(test_s0, t_eval, hcw_rhs)
        plain = propagate(test_s0, t_eval, residual_rhs_from_net(net_plain))
        reg = propagate(test_s0, t_eval, residual_rhs_from_net(net_reg))

    err_base = torch.linalg.norm(base[:, :, :2] - truth[:, :, :2], dim=-1).mean(dim=0)
    err_plain = torch.linalg.norm(plain[:, :, :2] - truth[:, :, :2], dim=-1).mean(dim=0)
    err_reg = torch.linalg.norm(reg[:, :, :2] - truth[:, :, :2], dim=-1).mean(dim=0)
    print(f"  72h mean drift | HCW={err_base[-1]:.1f} m, lambda=0={err_plain[-1]:.1f} m, lambda=0.01={err_reg[-1]:.1f} m")
    print(f"  Regularized/plain final drift ratio: {float(err_reg[-1] / err_plain[-1]):.3f}")
    return {
        "t_hours": (t_eval.cpu().numpy() / 3600.0),
        "err_base": err_base.cpu().numpy(),
        "err_plain": err_plain.cpu().numpy(),
        "err_reg": err_reg.cpu().numpy(),
        "loss_plain": loss_plain,
        "loss_reg": loss_reg,
    }


def event_value_labels(states0: torch.Tensor, t_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    traj = propagate(states0, t_eval, true_rhs)
    dists = torch.linalg.norm(traj[:, :, :2], dim=-1)
    min_dist = dists.min(dim=-1).values
    signed_distance = min_dist - KEEP_OUT_M
    unsafe = signed_distance <= 0.0
    soft_event = -torch.logsumexp(-(dists - KEEP_OUT_M) / 120.0, dim=-1) * 120.0
    return unsafe.float(), signed_distance, soft_event


def train_proposer_models(
    train_x: torch.Tensor, unsafe_y: torch.Tensor, sdf_y: torch.Tensor, event_y: torch.Tensor
) -> tuple[ProposerMLP, ProposerMLP, ProposerMLP]:
    clf = ProposerMLP().to(DEVICE, DTYPE)
    sdf = ProposerMLP().to(DEVICE, DTYPE)
    event = ProposerMLP().to(DEVICE, DTYPE)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=4e-3)
    opt_sdf = torch.optim.Adam(sdf.parameters(), lr=4e-3)
    opt_event = torch.optim.Adam(event.parameters(), lr=4e-3)

    y_sdf = sdf_y / 1_000.0
    y_event = event_y / 1_000.0
    pos_weight = ((1.0 - unsafe_y).sum() / unsafe_y.sum().clamp_min(1.0)).clamp(1.0, 20.0)
    batch_size = 256
    for _ in range(260):
        perm = torch.randperm(train_x.shape[0], device=DEVICE)
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = train_x[idx]

            opt_clf.zero_grad()
            loss_clf = F.binary_cross_entropy_with_logits(
                clf(xb).squeeze(-1), unsafe_y[idx], pos_weight=pos_weight
            )
            loss_clf.backward()
            opt_clf.step()

            opt_sdf.zero_grad()
            loss_sdf = F.smooth_l1_loss(sdf(xb).squeeze(-1), y_sdf[idx])
            loss_sdf.backward()
            opt_sdf.step()

            opt_event.zero_grad()
            loss_event = F.smooth_l1_loss(event(xb).squeeze(-1), y_event[idx])
            loss_event.backward()
            opt_event.step()
    return clf, sdf, event


def screen_at_full_recall(scores: np.ndarray, true_unsafe: np.ndarray, high_is_risky: bool) -> dict[str, float]:
    n_unsafe = int(true_unsafe.sum())
    if n_unsafe == 0:
        threshold = float(scores.min() if high_is_risky else scores.max())
        flagged = np.zeros_like(true_unsafe, dtype=bool)
    elif high_is_risky:
        threshold = float(scores[true_unsafe].min())
        flagged = scores >= threshold
    else:
        threshold = float(scores[true_unsafe].max())
        flagged = scores <= threshold
    recall = 1.0 if n_unsafe == 0 else float((flagged & true_unsafe).sum() / n_unsafe)
    flagged_fraction = float(flagged.mean())
    return {"threshold": threshold, "recall": recall, "flagged_fraction": flagged_fraction}


def evaluate_value_fields() -> dict[str, object]:
    print("\n[Experiment 2] Safety value field proposers")
    t_eval = torch.arange(0.0, 6.0 * 3600.0 + 180.0, 180.0, dtype=DTYPE, device=DEVICE)
    train_x = sample_initial_states(1_200, seed=SEED + 3, near_conjunction=True)
    test_x = sample_initial_states(420, seed=SEED + 4, near_conjunction=True)
    unsafe_train, sdf_train, event_train = event_value_labels(train_x, t_eval)
    unsafe_test, sdf_test, event_test = event_value_labels(test_x, t_eval)
    print(f"  Unsafe cases: train={int(unsafe_train.sum())}/{len(train_x)}, test={int(unsafe_test.sum())}/{len(test_x)}")

    t0 = time.time()
    clf, sdf, event = train_proposer_models(train_x, unsafe_train, sdf_train, event_train)
    train_time = time.time() - t0
    with torch.no_grad():
        t_prop_start = time.time()
        clf_score = torch.sigmoid(clf(test_x).squeeze(-1)).cpu().numpy()
        sdf_score = (sdf(test_x).squeeze(-1) * 1_000.0).cpu().numpy()
        event_score = (event(test_x).squeeze(-1) * 1_000.0).cpu().numpy()
        proposer_time = time.time() - t_prop_start

    t_verify_start = time.time()
    _ = event_value_labels(test_x, t_eval)
    verifier_time = time.time() - t_verify_start
    verifier_per_case = verifier_time / len(test_x)
    true_unsafe = unsafe_test.cpu().numpy().astype(bool)

    baselines = {
        "Classifier": screen_at_full_recall(clf_score, true_unsafe, high_is_risky=True),
        "SDF": screen_at_full_recall(sdf_score, true_unsafe, high_is_risky=False),
        "Event-Value": screen_at_full_recall(event_score, true_unsafe, high_is_risky=False),
    }
    for name, result in baselines.items():
        pv_time = proposer_time / 3.0 + result["flagged_fraction"] * len(test_x) * verifier_per_case
        speedup = verifier_time / max(pv_time, 1e-9)
        result["speedup"] = float(speedup)
        print(
            f"  {name:11s} | threshold={result['threshold']:8.2f}, "
            f"flagged={100.0 * result['flagged_fraction']:5.1f}%, "
            f"recall={100.0 * result['recall']:5.1f}%, speedup={speedup:5.2f}x"
        )
    print(f"  Proposer training time: {train_time:.1f}s; verifier all-cases time: {verifier_time:.3f}s")
    return {
        "baselines": baselines,
        "unsafe_test": int(true_unsafe.sum()),
        "n_test": len(test_x),
        "verifier_time": verifier_time,
        "proposer_time": proposer_time,
    }


def train_helmholtz_model(states: torch.Tensor, epochs: int = 420) -> HelmholtzResidualModel:
    model = HelmholtzResidualModel().to(DEVICE, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-6)
    target_cons = conservative_residual(states) * ACCEL_SCALE
    target_drag = drag_residual(states) * ACCEL_SCALE
    target_total = target_cons + target_drag
    batch_size = 768
    for _ in range(epochs):
        perm = torch.randperm(states.shape[0], device=DEVICE)
        for start in range(0, states.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = states[idx]
            opt.zero_grad()
            pred_cons, pred_drag = model.components_scaled(xb)
            pred_total = pred_cons + pred_drag
            loss_total = F.mse_loss(pred_total, target_total[idx])
            loss_cons = 0.35 * F.mse_loss(pred_cons, target_cons[idx])
            loss_drag = 0.35 * F.mse_loss(pred_drag, target_drag[idx])
            loss = loss_total + loss_cons + loss_drag
            loss.backward()
            opt.step()
    model.eval()
    return model


def generate_maneuvers(t_eval: torch.Tensor) -> list[Burn]:
    rng = np.random.default_rng(SEED + 5)
    eligible = np.arange(25, len(t_eval) - 25)
    step_indices = sorted(rng.choice(eligible, size=5, replace=False).tolist())
    burns: list[Burn] = []
    for step_idx in step_indices:
        direction = rng.normal(size=2)
        direction = direction / np.linalg.norm(direction)
        mag = rng.uniform(0.045, 0.14)
        dv = torch.tensor(direction * mag, dtype=DTYPE, device=DEVICE).view(1, 2)
        burns.append(Burn(step_idx=step_idx, time_s=float(t_eval[step_idx].item()), delta_v=dv))
    return burns


def synthetic_observed_residuals(
    traj: torch.Tensor, t_eval: torch.Tensor, burns: list[Burn], dt: float
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    states = traj[:, 1:-1, :].reshape(-1, 4)
    observed = residual_accel(states).clone()
    times = t_eval[1:-1].cpu().numpy()
    for burn in burns:
        rel_idx = burn.step_idx - 1
        if 0 <= rel_idx < observed.shape[0]:
            observed[rel_idx] += burn.delta_v.reshape(2) / dt
    observed += torch.randn_like(observed) * 2.0e-7
    return states, observed, times


def evaluate_maneuver_isolation() -> dict[str, object]:
    print("\n[Experiment 3] Helmholtz residual separation and maneuver isolation")
    colloc_states, _ = build_collocation_dataset(14_000)
    t0 = time.time()
    model = train_helmholtz_model(colloc_states)
    print(f"  Helmholtz dual network trained in {time.time() - t0:.1f}s")

    t_eval = torch.arange(0.0, 18.0 * 3600.0 + 300.0, 300.0, dtype=DTYPE, device=DEVICE)
    s0 = torch.tensor(
        [[1_200.0, -800.0, 0.20, -2.0 * MEAN_MOTION * 1_200.0 + 0.05]],
        dtype=DTYPE,
        device=DEVICE,
    )
    burns = generate_maneuvers(t_eval)
    traj = propagate(s0, t_eval, true_rhs, maneuvers=burns)
    center_states, observed_res, times = synthetic_observed_residuals(traj, t_eval, burns, dt=300.0)

    pred_scaled = model(center_states)
    pred_res = pred_scaled / ACCEL_SCALE
    anomaly = torch.linalg.norm(observed_res - pred_res, dim=-1).detach().cpu().numpy()
    baseline = np.median(anomaly)
    mad = np.median(np.abs(anomaly - baseline)) + 1e-12
    threshold = max(float(baseline + 8.0 * mad), 2.5e-5)
    candidate_idx = np.where(anomaly > threshold)[0]
    detected: list[dict[str, float]] = []
    for idx in candidate_idx:
        if detected and abs(times[idx] - detected[-1]["time_s"]) < 450.0:
            if anomaly[idx] > detected[-1]["score"]:
                detected[-1] = {
                    "time_s": float(times[idx]),
                    "delta_v_mps": float(anomaly[idx] * 300.0),
                    "score": float(anomaly[idx]),
                }
            continue
        detected.append(
            {
                "time_s": float(times[idx]),
                "delta_v_mps": float(anomaly[idx] * 300.0),
                "score": float(anomaly[idx]),
            }
        )

    true_times = np.array([burn.time_s for burn in burns])
    matched = 0
    for burn in burns:
        if any(abs(item["time_s"] - burn.time_s) <= 450.0 for item in detected):
            matched += 1
    recall = matched / len(burns)
    print(f"  Skeptic threshold: {threshold:.3e} m/s^2")
    print(f"  Maneuver recall: {matched}/{len(burns)} = {100.0 * recall:.1f}%")
    print("  Recovered maneuvers:")
    for item in detected:
        print(f"    t={item['time_s'] / 3600.0:5.2f} h, delta-v~{item['delta_v_mps']:.4f} m/s")
    return {
        "times_h": times / 3600.0,
        "anomaly": anomaly,
        "threshold": threshold,
        "true_burns": burns,
        "detected": detected,
        "recall": recall,
        "true_times_h": true_times / 3600.0,
    }


def save_regularization_plot(results: dict[str, object]) -> Path:
    path = RESEARCH_DIR / "hcw_t97_regularization_drift.png"
    plt.figure(figsize=(9, 5))
    plt.plot(results["t_hours"], results["err_base"], "--", label="HCW base")
    plt.plot(results["t_hours"], results["err_plain"], label="Residual net, lambda=0")
    plt.plot(results["t_hours"], results["err_reg"], label="Residual net, lambda=0.01")
    plt.yscale("log")
    plt.xlabel("Rollout time (hours)")
    plt.ylabel("Mean position drift error (m)")
    plt.title("T-097 residual-gradient regularization ablation")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_value_plot(results: dict[str, object]) -> Path:
    path = RESEARCH_DIR / "hcw_t97_value_baselines.png"
    baselines = results["baselines"]
    names = list(baselines.keys())
    speedups = [baselines[name]["speedup"] for name in names]
    flagged = [100.0 * baselines[name]["flagged_fraction"] for name in names]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax1.bar(x - 0.18, speedups, width=0.36, color="tab:blue", label="Speedup at 100% recall")
    ax1.set_ylabel("Compute speedup (x)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, flagged, width=0.36, color="tab:orange", label="Verifier load")
    ax2.set_ylabel("Flagged for verifier (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("T-097 value-field proposer baselines")
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_maneuver_plot(results: dict[str, object]) -> Path:
    path = RESEARCH_DIR / "hcw_t97_maneuver_reconstruction.png"
    plt.figure(figsize=(10, 5))
    plt.plot(results["times_h"], results["anomaly"], label="Skeptic anomaly score")
    plt.axhline(results["threshold"], color="tab:red", linestyle="--", label="Detection threshold")
    for burn in results["true_burns"]:
        plt.axvline(burn.time_s / 3600.0, color="black", alpha=0.35, linewidth=1.2)
    det_times = [item["time_s"] / 3600.0 for item in results["detected"]]
    det_dv = [item["delta_v_mps"] for item in results["detected"]]
    if det_times:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.stem(det_times, det_dv, linefmt="tab:green", markerfmt="go", basefmt=" ", label="Recovered delta-v")
        ax2.set_ylabel("Recovered delta-v magnitude (m/s)")
    plt.xlabel("Trajectory time (hours)")
    plt.ylabel("Residual anomaly (m/s^2)")
    plt.title("T-097 maneuver isolation from residual anomalies")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def write_report(
    reg: dict[str, object],
    values: dict[str, object],
    maneuver: dict[str, object],
    paths: list[Path],
) -> Path:
    path = RESEARCH_DIR / "hcw_maneuver_report.md"
    baselines = values["baselines"]
    best_name = max(baselines.keys(), key=lambda name: baselines[name]["speedup"])
    detected_lines = "\n".join(
        f"- t={item['time_s'] / 3600.0:.2f} h, recovered |delta-v|={item['delta_v_mps']:.4f} m/s"
        for item in maneuver["detected"]
    )
    true_lines = "\n".join(
        f"- t={burn.time_s / 3600.0:.2f} h, true |delta-v|={burn.magnitude:.4f} m/s"
        for burn in maneuver["true_burns"]
    )
    report = f"""# T-097 Residual Regularization, Safety Value Fields & Maneuver Isolation

## Setup

This validation uses 2D planar HCW relative motion at a 500 km LEO reference
orbit. The base dynamics are linear HCW. The synthetic truth model adds:

- a conservative position-dependent cubic gravity anomaly,
  `a_c(r) = -{C_J2_CUBIC:.2e} r^3`;
- a non-conservative quadratic drag term aligned with anti-velocity,
  `a_d(v) = -{C_DRAG:.2e} ||v|| v`;
- sporadic active thrust impulses injected as instantaneous delta-v burns.

All neural fields are trained on collocation states and then evaluated through
rollout or proposer-verifier experiments.

## Experiment 1: Residual-Gradient Regularization

The physics-anchored Neural ODE predicts only residual acceleration:

`d/dt [r, v] = [v, a_HCW(r, v) + a_neural(r, v)]`.

The ablation compares standard residual fitting with an added Jacobian penalty:

`L = MSE(a_neural, a_residual) + lambda ||grad_r a_neural||_F^2`.

Results on 72 hour rollouts:

- HCW-only final mean position drift: {reg['err_base'][-1]:.2f} m
- lambda=0 final mean position drift: {reg['err_plain'][-1]:.2f} m
- lambda=0.01 final mean position drift: {reg['err_reg'][-1]:.2f} m
- regularized/plain final drift ratio: {float(reg['err_reg'][-1] / max(reg['err_plain'][-1], 1e-9)):.3f}

The regularization plot is `{paths[0].name}`.

## Experiment 2: Safety Value Field Baselines

Three proposer models were trained from initial relative states:

1. Conjunction classifier with binary cross-entropy.
2. Signed distance field regressor predicting `min_t ||r(t)|| - r_keepout`.
3. Event-value regressor using a soft-min continuous event value.

Each proposer was calibrated to send enough cases to the exact rollout verifier
to achieve 100% safety recall on the test set.

| Model | Threshold | Verifier load | Recall | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Classifier | {baselines['Classifier']['threshold']:.3f} | {100.0 * baselines['Classifier']['flagged_fraction']:.1f}% | {100.0 * baselines['Classifier']['recall']:.1f}% | {baselines['Classifier']['speedup']:.2f}x |
| SDF | {baselines['SDF']['threshold']:.3f} m | {100.0 * baselines['SDF']['flagged_fraction']:.1f}% | {100.0 * baselines['SDF']['recall']:.1f}% | {baselines['SDF']['speedup']:.2f}x |
| Event-Value | {baselines['Event-Value']['threshold']:.3f} m | {100.0 * baselines['Event-Value']['flagged_fraction']:.1f}% | {100.0 * baselines['Event-Value']['recall']:.1f}% | {baselines['Event-Value']['speedup']:.2f}x |

Best speedup at 100% recall: **{best_name}**, {baselines[best_name]['speedup']:.2f}x.
The value baseline plot is `{paths[1].name}`.

## Experiment 3: Maneuver & Drag Separation

The Helmholtz-style residual model uses two constrained components:

- a potential network `Phi(r)` whose conservative acceleration is
  `-grad_r Phi(r)`;
- a dissipative network that predicts a non-negative drag magnitude and projects
  it strictly onto the anti-velocity direction.

The Skeptic Agent computes:

`||a_obs - (a_base - grad Phi + f_d)||`

and isolates samples above an adaptive median/MAD threshold as active maneuvers.

True maneuvers:

{true_lines}

Recovered maneuver candidates:

{detected_lines}

Maneuver recall: {100.0 * maneuver['recall']:.1f}%.
The reconstruction plot is `{paths[2].name}`.

## Discussion

The experiment demonstrates the Phase 2 mechanics required for T-097: smoother
residual-field training, value-field screening before expensive verification,
and separation of smooth conservative/dissipative dynamics from impulsive
maneuvers. The setup is synthetic and deterministic, so the reported metrics are
validation evidence for the implementation path rather than a claim about
real ESA Kelvins operational performance.
"""
    path.write_text(report)
    return path


def verify_artifacts(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty artifacts: {missing}")


def main() -> None:
    set_deterministic()
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("T-097 HCW RESIDUAL REGULARIZATION, VALUE FIELDS, MANEUVER ISOLATION")
    print("=" * 80)
    print(f"Mean motion: {MEAN_MOTION:.6e} rad/s")

    reg_results = evaluate_regularization()
    value_results = evaluate_value_fields()
    maneuver_results = evaluate_maneuver_isolation()

    plot_paths = [
        save_regularization_plot(reg_results),
        save_value_plot(value_results),
        save_maneuver_plot(maneuver_results),
    ]
    report_path = write_report(reg_results, value_results, maneuver_results, plot_paths)
    all_paths = plot_paths + [report_path]
    verify_artifacts(all_paths)

    print("\nArtifacts verified on disk:")
    for path in all_paths:
        print(f"  {path} ({path.stat().st_size} bytes)")
    print("=" * 80)
    print("T-097 VERIFICATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
