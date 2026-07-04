import pytest
import torch
import torch.nn as nn
import numpy as np

from services.ml.node_model import (
    ResidualAccelerationNet, DifferentiablePotentialMLP,
    GeometricResidualODE, GeometricResidualPropagator,
    SundmanNeuralODEPropagator
)
from services.ml.sampler import TrajectoryTubeSampler
from services.ml.maneuver import HelmholtzManeuverDetector
from services.ml.conjunction import ConjunctionValueMLP, ProposerVerifierScreening

class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

@pytest.fixture
def base_components():
    device = torch.device("cpu")
    potential_net = DifferentiablePotentialMLP(hidden_dim=8).double().to(device)
    drag_net = ResidualAccelerationNet(hidden_dim=8, out_dim=1).double().to(device)
    
    geom_vf = GeometricResidualODE(
        potential_net=potential_net, drag_net=drag_net,
        e_net=ZeroNet(), b_net=ZeroNet(),
        bstar=0.0, use_gravity=True, use_j2=True, use_drag=False
    ).to(device)
    
    base_prop = GeometricResidualPropagator(geom_vf, rtol=1e-5, atol=1e-7, method="rk4").double().to(device)
    return base_prop, geom_vf

def test_sundman_propagator(base_components):
    base_prop, geom_vf = base_components
    device = torch.device("cpu")
    
    sundman_prop = SundmanNeuralODEPropagator(geom_vf, rtol=1e-5, atol=1e-7, method="rk4").double().to(device)
    
    pos0 = torch.tensor([6878136.3, 0.0, 0.0], dtype=torch.float64, device=device)
    vel0 = torch.tensor([0.0, 7500.0, 0.0], dtype=torch.float64, device=device)
    state0 = torch.cat([pos0, vel0]).unsqueeze(0)  # (1, 6)
    
    s_span = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64, device=device)
    traj_7d = sundman_prop(state0, s_span)
    
    assert traj_7d.shape == (1, 3, 7)
    # Check that time state (7th element) increases monotonically
    assert traj_7d[0, 2, 6] > traj_7d[0, 0, 6]

def test_trajectory_tube_sampler(base_components):
    base_prop, geom_vf = base_components
    device = torch.device("cpu")
    
    sampler = TrajectoryTubeSampler(geom_vf, q_process_noise=1e-11).to(device)
    
    pos0 = torch.tensor([6878136.3, 0.0, 0.0], dtype=torch.float64, device=device)
    vel0 = torch.tensor([0.0, 7500.0, 0.0], dtype=torch.float64, device=device)
    states = torch.cat([pos0, vel0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 6)
    
    P0 = torch.eye(6, dtype=torch.float64, device=device).unsqueeze(0) * 1e-4  # (1, 6, 6)
    t_eval = torch.tensor([0.0], dtype=torch.float64, device=device)
    
    P_seq = sampler.propagate_covariance_stm(states, t_eval, P0)
    assert P_seq.shape == (1, 1, 6, 6)
    
    colloc = sampler.generate_collocation_points(states, P_seq, num_samples_per_step=5)
    assert colloc.shape == (1, 1, 5, 6)

def test_helmholtz_maneuver_detector():
    device = torch.device("cpu")
    # Mock Helmholtz model that outputs zeros
    class MockHelmholtzModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3, dtype=torch.float64, device=device)
            
    mock_model = MockHelmholtzModel()
    detector = HelmholtzManeuverDetector(mock_model, n_mad=3.0, min_threshold=1e-5)
    
    states = torch.randn(5, 6, dtype=torch.float64, device=device)
    # Step 2 has a large anomaly
    observed_acc = torch.randn(5, 3, dtype=torch.float64, device=device) * 1e-7
    observed_acc[2] += torch.tensor([1.0e-3, 0.0, 0.0], dtype=torch.float64, device=device)
    
    times = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float64, device=device)
    
    detections = detector.detect_maneuvers(states, observed_acc, times, dt=10.0)
    
    assert len(detections) == 1
    assert abs(detections[0]["time_s"] - 20.0) < 1e-3
    assert detections[0]["delta_v_mps"] > 1e-3

def test_conjunction_screening(base_components):
    base_prop, _ = base_components
    device = torch.device("cpu")
    
    proposer = ConjunctionValueMLP(in_dim=6, hidden_dim=8).double().to(device)
    pipeline = ProposerVerifierScreening(proposer, base_prop, r_keepout=100.0)
    
    pos0 = torch.tensor([6878136.3, 0.0, 0.0], dtype=torch.float64, device=device)
    vel0 = torch.tensor([0.0, 7500.0, 0.0], dtype=torch.float64, device=device)
    s0_batch = torch.cat([pos0, vel0]).unsqueeze(0)  # (1, 6)
    
    t_eval = torch.tensor([0.0, 10.0], dtype=torch.float64, device=device)
    
    # Mock run of screening
    results = pipeline.run_screening(s0_batch, t_eval, screening_threshold=200.0)
    
    assert "flagged_mask" in results
    assert "pipeline_time" in results
    assert "speedup" in results
    assert len(results["predicted_values"]) == 1
