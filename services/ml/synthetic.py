"""Synthetic orbital data generator for foundation model training.

Generates labeled satellite trajectories with injected events:
  - Normal station-keeping (baseline)
  - Maneuvers (delta-v at known times)
  - Decay (drag increase)
  - Breakup (sudden eccentricity/inclination change)

Each trajectory is a sequence of synthetic "TLE-like" observations:
  [epoch, mean_motion, eccentricity, inclination, bstar, alt_km]
with ground-truth labels at each timestep.

The physics engine is our own propagator (J2+J3+J4+drag). Measurement
noise is added to simulate TLE fitting uncertainty.

Usage:
    python -m services.ml.synthetic --count 10000 --output data/synthetic/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.brain.dynamics import (
    propagate_state,
    MU_EARTH,
    R_EARTH,
)


# ── Orbit generation ──


def _random_leo_state(rng: np.random.Generator) -> np.ndarray:
    """Generate a random LEO circular orbit state vector."""
    alt_km = rng.uniform(300, 800)
    r = R_EARTH + alt_km * 1000
    v = np.sqrt(MU_EARTH / r)

    incl = rng.uniform(40, 100) * np.pi / 180
    raan = rng.uniform(0, 360) * np.pi / 180
    arg = rng.uniform(0, 360) * np.pi / 180

    # Position in orbital plane
    r_orb = np.array([r * np.cos(arg), r * np.sin(arg), 0.0])
    v_orb = np.array([-v * np.sin(arg), v * np.cos(arg), 0.0])

    # Rotation to ECI (simplified: RAAN + inclination)
    ci, si = np.cos(incl), np.sin(incl)
    co, so = np.cos(raan), np.sin(raan)
    R_mat = np.array(
        [
            [co, -so * ci, so * si],
            [so, co * ci, -co * si],
            [0, si, ci],
        ]
    )

    pos = R_mat @ r_orb
    vel = R_mat @ v_orb
    return np.concatenate([pos, vel])


def _state_to_elements(state: np.ndarray) -> dict:
    """Convert state vector to TLE-like orbital elements."""
    pos = state[:3]
    vel = state[3:]
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    alt_km = (r - R_EARTH) / 1000

    # Specific energy → semi-major axis
    energy = 0.5 * v**2 - MU_EARTH / r
    if energy >= 0:
        a = r  # fallback for escape
    else:
        a = -MU_EARTH / (2 * energy)

    # Angular momentum
    h = np.cross(pos, vel)
    h_mag = np.linalg.norm(h)

    # Inclination
    incl = np.arccos(np.clip(h[2] / h_mag, -1, 1)) * 180 / np.pi

    # Eccentricity
    e_vec = np.cross(vel, h) / MU_EARTH - pos / r
    ecc = np.linalg.norm(e_vec)

    # Mean motion (rev/day)
    if a > 0:
        n = np.sqrt(MU_EARTH / a**3)  # rad/s
        mm = n * 86400 / (2 * np.pi)  # rev/day
    else:
        mm = 15.0  # fallback

    return {
        "mean_motion": mm,
        "eccentricity": ecc,
        "inclination": incl,
        "alt_km": alt_km,
    }


# ── Event injection ──


def _inject_maneuver(
    state: np.ndarray, rng: np.random.Generator, magnitude: float = 0.0
) -> np.ndarray:
    """Apply a delta-v to the velocity vector."""
    if magnitude <= 0:
        magnitude = rng.uniform(0.1, 5.0)  # m/s

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    new_state = state.copy()
    new_state[3:] += magnitude * direction
    return new_state


def _inject_breakup(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a breakup: large random delta-v (10-100 m/s)."""
    magnitude = rng.uniform(10, 100)
    return _inject_maneuver(state, rng, magnitude)


# ── Trajectory generator ──

LABEL_NORMAL = 0
LABEL_MANEUVER = 1
LABEL_DECAY = 2
LABEL_BREAKUP = 3

LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}


def generate_trajectory(
    rng: np.random.Generator,
    n_steps: int = 100,
    dt_hours: float = 8.0,
    event_type: str = "normal",
    noise_mm: float = 0.005,
    noise_ecc: float = 0.0002,
    noise_incl: float = 0.01,
) -> dict:
    """Generate one synthetic satellite trajectory with optional event.

    Returns:
        {
            "elements": np.array shape (n_steps, 6),
                columns: [epoch_h, mm, ecc, incl, bstar, alt_km]
            "labels": np.array shape (n_steps,),
                0=normal, 1=maneuver, 2=decay, 3=breakup
            "event_type": str,
            "event_step": int or None,
            "metadata": dict
        }
    """
    state = _random_leo_state(rng)
    bstar = rng.uniform(1e-5, 5e-3) * rng.choice([1, -1])
    dt_s = dt_hours * 3600

    # Decide event timing
    if event_type == "normal":
        event_step = None
    else:
        event_step = rng.integers(n_steps // 4, 3 * n_steps // 4)

    elements = np.zeros((n_steps, 6))
    labels = np.zeros(n_steps, dtype=np.int32)

    decay_active = False

    for i in range(n_steps):
        # Propagate
        if i > 0:
            state, ok = propagate_state(state, dt_s, bstar=bstar)
            if not ok:
                break

        # Inject events
        if event_step is not None and i == event_step:
            if event_type == "maneuver":
                state = _inject_maneuver(state, rng)
            elif event_type == "breakup":
                state = _inject_breakup(state, rng)
            elif event_type == "decay":
                decay_active = True
                bstar = abs(bstar) * rng.uniform(3, 10)  # increase drag

        # Label
        if event_type == "maneuver" and event_step is not None and i >= event_step:
            labels[i] = LABEL_MANEUVER
        elif event_type == "breakup" and event_step is not None and i >= event_step:
            labels[i] = LABEL_BREAKUP
        elif decay_active:
            labels[i] = LABEL_DECAY
        else:
            labels[i] = LABEL_NORMAL

        # Convert to elements + add noise
        elems = _state_to_elements(state)
        elements[i] = [
            i * dt_hours,
            elems["mean_motion"] + rng.normal(0, noise_mm),
            max(0, elems["eccentricity"] + rng.normal(0, noise_ecc)),
            elems["inclination"] + rng.normal(0, noise_incl),
            bstar,
            elems["alt_km"],
        ]

    return {
        "elements": elements,
        "labels": labels,
        "event_type": event_type,
        "event_step": event_step,
        "metadata": {
            "bstar_initial": bstar,
            "n_steps": n_steps,
            "dt_hours": dt_hours,
        },
    }


def generate_dataset(
    count: int = 10000,
    n_steps: int = 100,
    seed: int = 42,
    event_mix: dict | None = None,
) -> dict:
    """Generate a full training dataset.

    Args:
        count: number of trajectories
        n_steps: timesteps per trajectory
        seed: random seed
        event_mix: fraction of each event type,
            default: {"normal": 0.5, "maneuver": 0.2, "decay": 0.15, "breakup": 0.15}

    Returns:
        {
            "X": np.array (count, n_steps, 6),
            "y": np.array (count, n_steps),
            "event_types": list[str],
            "event_steps": list[int|None],
        }
    """
    if event_mix is None:
        event_mix = {"normal": 0.5, "maneuver": 0.2, "decay": 0.15, "breakup": 0.15}

    rng = np.random.default_rng(seed)

    # Build event schedule
    events = []
    for etype, frac in event_mix.items():
        events.extend([etype] * int(count * frac))
    while len(events) < count:
        events.append("normal")
    rng.shuffle(events)

    X = np.zeros((count, n_steps, 6))
    y = np.zeros((count, n_steps), dtype=np.int32)
    event_types = []
    event_steps = []

    t0 = time.time()
    for i in range(count):
        traj = generate_trajectory(rng, n_steps=n_steps, event_type=events[i])
        X[i] = traj["elements"]
        y[i] = traj["labels"]
        event_types.append(traj["event_type"])
        event_steps.append(traj["event_step"])

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (count - i - 1) / rate
            print(f"  [{i + 1}/{count}] {rate:.0f} traj/s, ETA {eta:.0f}s")

    return {
        "X": X,
        "y": y,
        "event_types": event_types,
        "event_steps": event_steps,
    }


# ── CLI ──


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic orbital trajectories for ML training.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of trajectories (default: 10000)",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Timesteps per trajectory (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("data/synthetic"), help="Output directory"
    )
    args = parser.parse_args(argv)

    print(f"Generating {args.count} trajectories × {args.steps} steps...")
    dataset = generate_dataset(count=args.count, n_steps=args.steps, seed=args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "X.npy", dataset["X"])
    np.save(args.output / "y.npy", dataset["y"])

    # Stats
    from collections import Counter

    type_counts = Counter(dataset["event_types"])
    print(f"\nDataset shape: X={dataset['X'].shape}, y={dataset['y'].shape}")
    print(f"Event mix: {dict(type_counts)}")
    print(f"Saved to {args.output}/")

    size_mb = (dataset["X"].nbytes + dataset["y"].nbytes) / 1e6
    print(f"Size: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
