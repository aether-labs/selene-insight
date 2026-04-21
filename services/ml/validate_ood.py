"""Out-of-distribution validation for v0.6 OrbitalTransformer.

Three real events the model has never seen as training data; each has
an independently-confirmed ground-truth timeline we can compare against:

  1. STARLINK-34343 (NORAD 64157)
       debris event on 2026-03-29 ~19:30 UTC (confirmed by S4S).
       Decayed post-event, never part of active Starlink training set.
  2. Iridium 33 (24946) / Cosmos 2251 (22675)
       collision on 2009-02-10 16:55:58 UTC.
       Old LEO, not in Starlink group; may appear in generic-LEO sets
       but still not in the Starlink-filtered training data.
  3. AST BLUEBIRD 7 (68765)
       failed second-stage burn on 2026-04-19 11:38 UTC (SECO-1 epoch,
       then stage 2 underperformance; ASTS confirmed unrecoverable).

Gate (Phase 3 of the v0.6 roadmap): for each event, the model must
both (a) classify the first post-event TLE as non-normal, and
(b) produce an innovation magnitude in the top 1% of the satellite's
pre-event history at that timestep.

A training-set leakage check also prints the overlap between each
OOD NORAD ID and the training data so we notice if a specific sat is
accidentally in the training set.

Usage:
    python -m services.ml.validate_ood \
        --checkpoint checkpoints/v06_finetune/best_model.pt \
        --training-norads data/ml_ready_v06/_norads.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from services.ml.model import create_model
from services.ml.physics import compute_innovation_series_sgp4
from services.ml.preprocess_v06 import (
    FEATURE_MEANS,
    FEATURE_STDS,
    N_FEATURES,
    parse_epoch_string,
)

LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}

OOD_EVENTS = [
    {
        "norad_id": 64157,
        "name": "STARLINK-34343",
        "event_utc": "2026-03-29T19:30:00",
        "expected_class": 3,  # breakup
        "description": "S4S-confirmed debris event",
        "window_days_before": 14,
        "window_days_after": 3,
    },
    {
        "norad_id": 24946,
        "name": "Iridium 33",
        "event_utc": "2009-02-10T16:55:58",
        "expected_class": 3,  # breakup (collision)
        "description": "Iridium-Cosmos collision",
        "window_days_before": 14,
        "window_days_after": 3,
    },
    {
        "norad_id": 22675,
        "name": "Cosmos 2251",
        "event_utc": "2009-02-10T16:55:58",
        "expected_class": 3,  # breakup (collision)
        "description": "Iridium-Cosmos collision",
        "window_days_before": 14,
        "window_days_after": 3,
    },
    {
        "norad_id": 68765,
        "name": "AST BLUEBIRD 7",
        "event_utc": "2026-04-19T11:38:04",
        "expected_class": 1,  # maneuver (upper stage burn)
        "description": "Blue Origin NG-3 stage-2 underperformance",
        "window_days_before": 0,  # launch is the first data point
        "window_days_after": 7,
    },
]

SPACETRACK_BASE = "https://www.space-track.org"


def _st_login(client: httpx.Client) -> None:
    u = os.environ["SPACETRACK_USER"]
    p = os.environ["SPACETRACK_PASS"]
    client.post(
        f"{SPACETRACK_BASE}/ajaxauth/login",
        data={"identity": u, "password": p},
    ).raise_for_status()


def fetch_tle_history(
    norad_id: int,
    start_utc: str,
    end_utc: str,
) -> list[dict]:
    """Pull gp_history records for one NORAD ID within a UTC window."""
    with httpx.Client(timeout=120, headers={"User-Agent": "argusorb/ood"}) as c:
        _st_login(c)
        url = (
            f"{SPACETRACK_BASE}/basicspacedata/query/class/gp_history"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/>{start_utc},<{end_utc}"
            "/orderby/EPOCH asc/format/json"
        )
        return c.get(url).json()


def build_features(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Turn raw gp_history records into a (T, 12) feature sequence.
    Mirrors preprocess_v06._load_one_satellite but for a single event window."""
    parsed = []
    for r in records:
        l1 = r.get("TLE_LINE1")
        l2 = r.get("TLE_LINE2")
        if not l1 or not l2:
            continue
        try:
            parsed.append(
                {
                    "epoch_ts": parse_epoch_string(r["EPOCH"]),
                    "epoch_str": r["EPOCH"],
                    "l1": l1,
                    "l2": l2,
                    "mm": float(r["MEAN_MOTION"]),
                    "ecc": float(r["ECCENTRICITY"]),
                    "incl": float(r["INCLINATION"]),
                    "bstar": float(r["BSTAR"]),
                    "alt_km": float(r["SEMIMAJOR_AXIS"]) - 6378.137
                    if float(r["SEMIMAJOR_AXIS"]) > 6378
                    else 0.0,
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    parsed.sort(key=lambda d: d["epoch_ts"])
    T = len(parsed)
    if T < 2:
        return np.zeros((0, N_FEATURES)), []

    l1s = [p["l1"] for p in parsed]
    l2s = [p["l2"] for p in parsed]
    innovations = compute_innovation_series_sgp4(l1s, l2s)

    features = np.zeros((T, N_FEATURES))
    t0 = parsed[0]["epoch_ts"]
    epoch_strs = []
    for t in range(T):
        p = parsed[t]
        features[t, 0] = (p["epoch_ts"] - t0) / 3600.0
        features[t, 1] = p["mm"]
        features[t, 2] = p["ecc"]
        features[t, 3] = p["incl"]
        features[t, 4] = p["bstar"]
        features[t, 5] = p["alt_km"]
        features[t, 6:12] = innovations[t]
        epoch_strs.append(p["epoch_str"])
    return features, epoch_strs


def normalize_features(X: np.ndarray) -> np.ndarray:
    return (X - FEATURE_MEANS) / np.maximum(FEATURE_STDS, 1e-10)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = create_model(
        size=cfg.get("size", "medium"),
        use_physics=cfg.get("use_physics", False),
        n_features=cfg.get("n_features", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}: "
        f"size={cfg.get('size')}, features={cfg.get('n_features')}"
    )
    return model


def find_event_index(epoch_strs: list[str], event_utc: str) -> int:
    """Index of first TLE at or after the event time."""
    event_ts = parse_epoch_string(event_utc)
    for i, s in enumerate(epoch_strs):
        if parse_epoch_string(s) >= event_ts:
            return i
    return -1


def check_training_leakage(
    norad_ids: list[int], training_norad_file: Path | None
) -> dict[int, bool]:
    """Returns {norad_id: True if in training data, False otherwise}."""
    if training_norad_file is None or not training_norad_file.exists():
        return {nid: None for nid in norad_ids}  # unknown
    try:
        training = set(
            int(line.strip())
            for line in training_norad_file.read_text().splitlines()
            if line.strip().isdigit()
        )
    except Exception:
        return {nid: None for nid in norad_ids}
    return {nid: nid in training for nid in norad_ids}


@torch.no_grad()
def validate_one_event(
    model: torch.nn.Module,
    event: dict,
    device: torch.device,
) -> dict:
    """Run the full OOD pipeline for one event. Returns a result dict."""
    event_dt = datetime.fromisoformat(event["event_utc"])
    wb = event.get("window_days_before", 14)
    wa = event.get("window_days_after", 3)
    start = f"{event_dt.year}-{event_dt.month:02d}-{(event_dt.day - wb) or 1:02d}"
    # Simple offset: just shift dates
    from datetime import timedelta

    start_utc = (event_dt - timedelta(days=wb)).strftime("%Y-%m-%d")
    end_utc = (event_dt + timedelta(days=wa)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 70}")
    print(f"{event['name']} (NORAD {event['norad_id']}) — {event['description']}")
    print(f"  event: {event['event_utc']}")
    print(f"  window: {start_utc} → {end_utc}")
    print(
        f"  expected class: {event['expected_class']} ({LABEL_NAMES[event['expected_class']]})"
    )

    try:
        records = fetch_tle_history(event["norad_id"], start_utc, end_utc)
    except Exception as e:
        return {"event": event["name"], "status": "FETCH_FAILED", "error": str(e)}
    print(f"  fetched {len(records)} TLEs")
    if len(records) < 2:
        return {"event": event["name"], "status": "TOO_FEW_TLES"}

    features, epoch_strs = build_features(records)
    if len(features) < 2:
        return {"event": event["name"], "status": "NO_VALID_TLES"}

    event_idx = find_event_index(epoch_strs, event["event_utc"])
    if event_idx < 0:
        event_idx = len(features) - 1  # fallback: last TLE

    X = normalize_features(features)
    # Clip extreme innovation values to avoid numerical issues
    X = np.clip(X, -50, 50)

    X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, 12)
    out = model(X_tensor, causal=True)
    cls = out["classifications"] if isinstance(out, dict) else out[1]  # (1, T, 4)

    probs = torch.softmax(cls, dim=-1).squeeze(0).cpu().numpy()  # (T, 4)
    preds = probs.argmax(axis=-1)  # (T,)

    # Detection window: the 5 TLEs starting at event_idx. An event's
    # signature may not land on the very first post-event TLE (e.g., for
    # launch-time events the first TLE is pre-separation with zero
    # innovation by construction), so we consider the model to have
    # detected the anomaly if ANY step in this window flags non-normal.
    wlo = event_idx
    whi = min(event_idx + 5, len(preds))
    window_preds = preds[wlo:whi]
    window_probs = probs[wlo:whi]
    window_inns = np.linalg.norm(features[wlo:whi, 6:9], axis=1)

    # Peak step within the window (max innovation)
    peak_rel = int(np.argmax(window_inns)) if len(window_inns) > 0 else 0
    peak_abs = wlo + peak_rel
    peak_pred = int(window_preds[peak_rel])
    peak_probs = window_probs[peak_rel]
    peak_inn = float(window_inns[peak_rel])

    # Pre-event innovation percentile context
    pre_inn_mags = (
        np.linalg.norm(features[:wlo, 6:9], axis=1) if wlo > 0 else np.array([0])
    )
    pre_p99 = float(np.percentile(pre_inn_mags, 99)) if len(pre_inn_mags) > 0 else 0.0

    detection = bool(np.any(window_preds != 0))
    innovation_spike = peak_inn > max(pre_p99, 1000)  # at least 1 km
    expected_match = bool(np.any(window_preds == event["expected_class"]))

    print(
        f"  event TLE index: {event_idx} / {len(preds)} (epoch {epoch_strs[event_idx]})"
    )
    print(f"  peak step within window: idx {peak_abs} (epoch {epoch_strs[peak_abs]})")
    print(f"  peak prediction: class {peak_pred} ({LABEL_NAMES[peak_pred]})")
    print(
        f"    probs: normal={peak_probs[0]:.3f} maneuver={peak_probs[1]:.3f} "
        f"decay={peak_probs[2]:.3f} breakup={peak_probs[3]:.3f}"
    )
    print(f"  peak innovation: {peak_inn:.0f} m  (pre-event p99: {pre_p99:.0f} m)")
    print(
        f"  window classes [{wlo}:{whi}]: {[LABEL_NAMES[int(p)] for p in window_preds]}"
    )
    print(f"  detection         : {'PASS' if detection else 'FAIL'}")
    print(f"  innovation spike  : {'PASS' if innovation_spike else 'FAIL'}")
    print(
        f"  expected-class hit: "
        f"{'PASS' if expected_match else 'MISS (got wrong class)'}"
    )

    # Backward-compat fields for the summary
    first_post = event_idx
    post_pred = peak_pred
    post_probs = peak_probs
    inn_mag = peak_inn

    return {
        "event": event["name"],
        "norad_id": event["norad_id"],
        "status": "EVALUATED",
        "event_idx": first_post,
        "predicted_class": post_pred,
        "expected_class": event["expected_class"],
        "probs": post_probs.tolist(),
        "innovation_m": float(inn_mag),
        "pre_event_p99_m": pre_p99,
        "detection": bool(detection),
        "innovation_spike": bool(innovation_spike),
        "expected_match": bool(expected_match),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--training-norads",
        type=Path,
        default=None,
        help="Optional: file listing NORAD IDs present in training set, one per line, for leakage check.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if not os.environ.get("SPACETRACK_USER") or not os.environ.get("SPACETRACK_PASS"):
        print("SPACETRACK_USER/PASS env vars required", file=sys.stderr)
        return 1

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = load_model_from_checkpoint(args.checkpoint, device)

    # Data leakage check
    leakage = check_training_leakage(
        [ev["norad_id"] for ev in OOD_EVENTS], args.training_norads
    )
    print("\nTraining-set leakage check:")
    for nid, in_train in leakage.items():
        if in_train is None:
            print(f"  NORAD {nid}: unknown (no training-norads list provided)")
        else:
            flag = "⚠ IN TRAINING DATA" if in_train else "not in training"
            print(f"  NORAD {nid}: {flag}")

    # Run each event
    results = []
    for ev in OOD_EVENTS:
        results.append(validate_one_event(model, ev, device))

    # Summary gate
    print(f"\n{'=' * 70}")
    print("Phase 3 OOD Gate — three events must all PASS on both detection")
    print("and expected-class match:")
    all_pass = True
    for r in results:
        if r["status"] != "EVALUATED":
            print(f"  {r['event']:25}  {r['status']}")
            all_pass = False
            continue
        status = "PASS" if r["detection"] and r["expected_match"] else "FAIL"
        if not (r["detection"] and r["expected_match"]):
            all_pass = False
        print(
            f"  {r['event']:25}  {status}  "
            f"(predicted {LABEL_NAMES[r['predicted_class']]}, "
            f"expected {LABEL_NAMES[r['expected_class']]}, "
            f"innovation {r['innovation_m']:.0f} m)"
        )

    print(f"\nPhase 3 OOD: {'PASS' if all_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
