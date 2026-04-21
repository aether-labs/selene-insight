"""v0.6 synthetic orbital data generator.

Changes from v0.5 generator:
  1. Output is (N, T, 12): 6 TLE-like elements + 6 Cartesian innovation
     channels (matches preprocess_v06 output schema for real data).
  2. Innovation is the truth-delta vs a "nominal" parallel track that
     runs the same physics without events. For normal trajectories this
     is exactly zero; for events it captures the event's cumulative
     state perturbation.
  3. Three new event types:
      - phasing_burn:          small single Δv (0.05 - 0.5 m/s)
      - multi_phase_maneuver:  2-3 Δv impulses spread across the window
      - attitude_anomaly:      B* oscillation (+200%/-150%) for 3-5
                               steps with orbital state unaffected;
                               signature lives in B* channel.
  4. Default count bumped to 100k for v0.6 training.

Labels stay 4-class to keep model architecture unchanged:
  NORMAL=0, MANEUVER=1, DECAY=2, BREAKUP=3.
The new event types are mapped into existing labels based on physical
similarity (see _event_to_label below).

Usage:
    python -m services.ml.synthetic_v06 --count 100000 \
        --output data/synthetic_v06/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.brain.dynamics import MU_EARTH, R_EARTH, propagate_state

# Labels kept identical to v0.5 to avoid model architecture changes
LABEL_NORMAL = 0
LABEL_MANEUVER = 1
LABEL_DECAY = 2
LABEL_BREAKUP = 3
LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}

# Event types (finer granularity than labels)
EVENT_TYPES = [
    "normal",
    "maneuver",  # 1-5 m/s single Δv
    "phasing_burn",  # 0.05-0.5 m/s single Δv
    "multi_phase_maneuver",  # 2-3 separate Δvs
    "decay",  # B* step increase (3-10×)
    "breakup",  # 10-100 m/s single Δv
    "attitude_anomaly",  # B* oscillation +200%/-150%
]


def _event_to_label(event_type: str) -> int:
    """Map fine-grained event type to the 4-class training label."""
    if event_type in ("maneuver", "phasing_burn", "multi_phase_maneuver"):
        return LABEL_MANEUVER
    if event_type == "breakup":
        return LABEL_BREAKUP
    if event_type in ("decay", "attitude_anomaly"):
        return LABEL_DECAY
    return LABEL_NORMAL


def _random_leo_state(rng: np.random.Generator) -> np.ndarray:
    """Generate a random LEO circular orbit state vector."""
    alt_km = rng.uniform(300, 800)
    r = R_EARTH + alt_km * 1000
    v = np.sqrt(MU_EARTH / r)

    incl = rng.uniform(40, 100) * np.pi / 180
    raan = rng.uniform(0, 360) * np.pi / 180
    arg = rng.uniform(0, 360) * np.pi / 180

    r_orb = np.array([r * np.cos(arg), r * np.sin(arg), 0.0])
    v_orb = np.array([-v * np.sin(arg), v * np.cos(arg), 0.0])

    ci, si = np.cos(incl), np.sin(incl)
    co, so = np.cos(raan), np.sin(raan)
    R_mat = np.array(
        [
            [co, -so * ci, so * si],
            [so, co * ci, -co * si],
            [0, si, ci],
        ]
    )
    return np.concatenate([R_mat @ r_orb, R_mat @ v_orb])


def _state_to_elements(state: np.ndarray) -> dict:
    """Convert state vector to TLE-like orbital elements."""
    pos, vel = state[:3], state[3:]
    r = float(np.linalg.norm(pos))
    v = float(np.linalg.norm(vel))
    alt_km = (r - R_EARTH) / 1000

    energy = 0.5 * v**2 - MU_EARTH / r
    a = r if energy >= 0 else -MU_EARTH / (2 * energy)

    h_vec = np.cross(pos, vel)
    h_mag = float(np.linalg.norm(h_vec))
    incl = float(np.arccos(np.clip(h_vec[2] / h_mag, -1, 1)) * 180 / np.pi)

    e_vec = np.cross(vel, h_vec) / MU_EARTH - pos / r
    ecc = float(np.linalg.norm(e_vec))
    mm = np.sqrt(MU_EARTH / a**3) * 86400 / (2 * np.pi) if a > 0 else 15.0
    return {
        "mean_motion": mm,
        "eccentricity": ecc,
        "inclination": incl,
        "alt_km": alt_km,
    }


def _apply_deltav(
    state: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Random-direction Δv in m/s applied to velocity."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    new_state = state.copy()
    new_state[3:] += magnitude * direction
    return new_state


def generate_trajectory(
    rng: np.random.Generator,
    n_steps: int = 100,
    dt_hours: float = 8.0,
    event_type: str = "normal",
    noise_mm: float = 0.005,
    noise_ecc: float = 0.0002,
    noise_incl: float = 0.01,
) -> dict:
    """Generate one trajectory with dual tracks (actual + nominal) so we
    can compute innovation = actual − nominal at every step.

    Returns:
      elements: (n_steps, 12) — [epoch_h, mm, ecc, incl, bstar, alt_km,
                                  inn_dx, inn_dy, inn_dz, inn_dvx, inn_dvy, inn_dvz]
      labels:   (n_steps,)
      ...
    """
    state_a = _random_leo_state(rng)
    state_n = state_a.copy()  # nominal track — diverges only when event fires
    bstar_a = rng.uniform(1e-5, 5e-3) * rng.choice([1, -1])
    bstar_n = bstar_a
    dt_s = dt_hours * 3600

    # Event schedule
    event_steps: list[int] = []
    if event_type == "normal":
        pass
    elif event_type == "multi_phase_maneuver":
        n_burns = int(rng.integers(2, 4))
        event_steps = sorted(
            set(
                int(rng.integers(n_steps // 6, 5 * n_steps // 6))
                for _ in range(n_burns)
            )
        )
    else:
        event_steps = [int(rng.integers(n_steps // 4, 3 * n_steps // 4))]

    # Attitude anomaly: B* oscillation window [start, end]
    attitude_window: tuple[int, int] | None = None
    if event_type == "attitude_anomaly" and event_steps:
        start = event_steps[0]
        duration = int(rng.integers(3, 6))
        attitude_window = (start, min(n_steps, start + duration))
        bstar_original = bstar_a

    elements = np.zeros((n_steps, 12))
    labels = np.zeros(n_steps, dtype=np.int32)
    label_val = _event_to_label(event_type)

    # Decay: persistent B* change applied once at first event step
    decay_active = False

    for i in range(n_steps):
        # Propagate both tracks one step
        if i > 0:
            state_a, ok_a = propagate_state(state_a, dt_s, bstar=bstar_a)
            state_n, ok_n = propagate_state(state_n, dt_s, bstar=bstar_n)
            if not (ok_a and ok_n):
                break

        # Apply events to the ACTUAL track only
        if i in event_steps:
            if event_type == "maneuver":
                state_a = _apply_deltav(state_a, rng.uniform(1, 5), rng)
            elif event_type == "phasing_burn":
                state_a = _apply_deltav(state_a, rng.uniform(0.05, 0.5), rng)
            elif event_type == "multi_phase_maneuver":
                state_a = _apply_deltav(state_a, rng.uniform(0.5, 3), rng)
            elif event_type == "breakup":
                state_a = _apply_deltav(state_a, rng.uniform(10, 100), rng)
            elif event_type == "decay" and not decay_active:
                decay_active = True
                bstar_a = abs(bstar_a) * rng.uniform(3, 10)

        # Attitude anomaly: oscillate B* only within the window
        if attitude_window is not None:
            start, end = attitude_window
            if start <= i < end:
                mid = (start + end) / 2
                # Rises to +200%, falls to -150% (relative), symmetric around mid
                phase = (i - start) / max(1, end - start - 1)
                if phase < 0.5:
                    bstar_a = bstar_original * (1 + 2.0 * (phase / 0.5))
                else:
                    bstar_a = bstar_original * (1 - 1.5 * ((phase - 0.5) / 0.5))
            elif i == end:
                bstar_a = bstar_original  # reset after window

        # Label the step
        if event_type == "normal":
            labels[i] = LABEL_NORMAL
        elif event_type in ("maneuver", "phasing_burn", "breakup"):
            if event_steps and i >= event_steps[0]:
                labels[i] = label_val
        elif event_type == "multi_phase_maneuver":
            if event_steps and i >= event_steps[0]:
                labels[i] = label_val
        elif event_type == "decay":
            if decay_active:
                labels[i] = label_val
        elif event_type == "attitude_anomaly":
            # Label persists past the B* window because the orbital state
            # has drifted from nominal — subsequent steps are not "normal"
            # even though B* is back to the original value.
            if attitude_window and i >= attitude_window[0]:
                labels[i] = label_val

        # Elements from actual state
        elems = _state_to_elements(state_a)
        # Innovation: actual − nominal (both in meters / m/s)
        innovation = state_a - state_n

        elements[i, 0] = i * dt_hours
        elements[i, 1] = elems["mean_motion"] + rng.normal(0, noise_mm)
        elements[i, 2] = max(0, elems["eccentricity"] + rng.normal(0, noise_ecc))
        elements[i, 3] = elems["inclination"] + rng.normal(0, noise_incl)
        elements[i, 4] = bstar_a
        elements[i, 5] = elems["alt_km"]
        elements[i, 6:12] = innovation

    return {
        "elements": elements,
        "labels": labels,
        "event_type": event_type,
        "event_steps": event_steps,
        "metadata": {"n_steps": n_steps, "dt_hours": dt_hours},
    }


DEFAULT_EVENT_MIX = {
    "normal": 0.40,
    "maneuver": 0.15,
    "phasing_burn": 0.10,
    "multi_phase_maneuver": 0.08,
    "decay": 0.12,
    "breakup": 0.10,
    "attitude_anomaly": 0.05,
}


def generate_dataset(
    count: int = 100000,
    n_steps: int = 100,
    seed: int = 42,
    event_mix: dict | None = None,
) -> dict:
    if event_mix is None:
        event_mix = DEFAULT_EVENT_MIX

    rng = np.random.default_rng(seed)
    events: list[str] = []
    for etype, frac in event_mix.items():
        events.extend([etype] * int(count * frac))
    while len(events) < count:
        events.append("normal")
    events = events[:count]
    rng.shuffle(events)

    X = np.zeros((count, n_steps, 12))
    y = np.zeros((count, n_steps), dtype=np.int32)
    event_types = []

    t0 = time.time()
    for i in range(count):
        traj = generate_trajectory(rng, n_steps=n_steps, event_type=events[i])
        X[i] = traj["elements"]
        y[i] = traj["labels"]
        event_types.append(traj["event_type"])

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (count - i - 1) / rate
            print(f"  [{i + 1}/{count}] {rate:.0f} traj/s, ETA {eta / 60:.1f} min")

    return {"X": X, "y": y, "event_types": event_types}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_v06"))
    args = parser.parse_args(argv)

    print(
        f"Generating {args.count:,} trajectories × {args.steps} steps (12 features)..."
    )
    ds = generate_dataset(count=args.count, n_steps=args.steps, seed=args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "X.npy", ds["X"])
    np.save(args.output / "y.npy", ds["y"])

    from collections import Counter

    tc = Counter(ds["event_types"])
    lc = Counter(ds["y"].flatten().tolist())
    print(f"\nShape: X={ds['X'].shape}, y={ds['y'].shape}")
    print(f"Event mix: {dict(tc)}")
    print(
        f"Label distribution: {{0:'normal', 1:'man', 2:'decay', 3:'breakup'}} → {dict(lc)}"
    )
    size_mb = (ds["X"].nbytes + ds["y"].nbytes) / 1e6
    print(f"Size: {size_mb:.1f} MB")
    print(f"Saved to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
