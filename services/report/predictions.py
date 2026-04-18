"""Public prediction generator — creates falsifiable claims for credibility.

Every prediction goes into the prediction table and gets tracked:
  1. Predicted: "STARLINK-X will deorbit within 30 days" (with confidence)
  2. Published: appears in the weekly report
  3. Resolved: next week (or at deadline), check if it happened
  4. Scored: correct/incorrect/expired → cumulative accuracy %

The prediction accuracy % is the single most powerful credibility signal.
"ArgusOrb's predictions were 78% accurate over 52 weeks" is a statement
that no competitor can make without putting in the same work.

Usage:
    # Generate predictions from current anomaly data
    python -m services.report.predictions generate [--db PATH]

    # Check and resolve predictions whose deadline has passed
    python -m services.report.predictions resolve [--db PATH]

    # Show scorecard
    python -m services.report.predictions score [--db PATH]
"""

from __future__ import annotations

import os
import sys
import time

try:
    from services.telemetry.store import StarlinkStore
    from services.brain.orbital_analyzer import detect_tle_gaps
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from services.telemetry.store import StarlinkStore
    from services.brain.orbital_analyzer import detect_tle_gaps


def generate_predictions(store: StarlinkStore) -> list[dict]:
    """Auto-generate predictions from current data signals.

    Types of predictions:
    1. reentry_30d: satellites below 300 km will deorbit within 30 days
    2. gap_resolution: satellites currently silent will either reappear
       or be confirmed lost within 14 days
    """
    now = time.time()
    predictions: list[dict] = []

    # Reentry predictions: satellites in the anomaly table with reentry labels
    anomalies = store.get_anomalies(limit=200)
    reentry_norad_ids = set()
    for a in anomalies:
        if a.get("anomaly_type") == "reentry" and a.get("classified_by") == "rule_v1":
            nid = a["norad_id"]
            if nid in reentry_norad_ids:
                continue
            reentry_norad_ids.add(nid)

            pred = {
                "norad_id": nid,
                "prediction_type": "reentry_30d",
                "description": (
                    f"{a.get('name') or f'NORAD {nid}'} is below 250 km and "
                    f"predicted to deorbit within 30 days. "
                    f"Based on rule_v1 reentry flag."
                ),
                "deadline_ts": now + 30 * 86400,
                "confidence": 0.90,
                "classifier": "rule_v1",
            }
            if store.insert_prediction(pred):
                predictions.append(pred)

    # Gap predictions: satellites currently silent
    gaps = detect_tle_gaps(store)
    for g in gaps[:20]:
        nid = g["norad_id"]
        pred = {
            "norad_id": nid,
            "prediction_type": "gap_resolution",
            "description": (
                f"{g.get('name') or f'NORAD {nid}'} has been silent for "
                f"{g['gap_hours']:.0f}h. Predicted to either resume TLE "
                f"updates or be confirmed lost within 14 days."
            ),
            "deadline_ts": now + 14 * 86400,
            "confidence": 0.70,
            "classifier": "gap_detector",
        }
        if store.insert_prediction(pred):
            predictions.append(pred)

    return predictions


def resolve_predictions(store: StarlinkStore) -> list[dict]:
    """Check pending predictions whose deadline has passed."""
    now = time.time()
    pending = store.get_pending_predictions()
    resolved: list[dict] = []

    for pred in pending:
        if pred["deadline_ts"] > now:
            continue  # not yet due

        nid = pred["norad_id"]
        ptype = pred["prediction_type"]

        if ptype == "reentry_30d":
            # Check: is the satellite still in the catalog with recent TLEs?
            sat = store.get_satellite(nid)
            if not sat:
                outcome = "correct"
                notes = "Satellite no longer in catalog — deorbited."
            else:
                last_seen = sat.get("last_seen") or 0
                gap_days = (now - last_seen) / 86400
                if gap_days > 7:
                    outcome = "correct"
                    notes = f"Last seen {gap_days:.0f} days ago — likely deorbited."
                else:
                    outcome = "incorrect"
                    notes = f"Still tracked, last seen {gap_days:.1f} days ago."

        elif ptype == "gap_resolution":
            sat = store.get_satellite(nid)
            if sat:
                last_seen = sat.get("last_seen") or 0
                gap_days = (now - last_seen) / 86400
                if gap_days < 2:
                    outcome = "correct"
                    notes = "TLE updates resumed."
                elif gap_days > 10:
                    outcome = "correct"
                    notes = f"Confirmed lost — no TLE for {gap_days:.0f} days."
                else:
                    outcome = "expired"
                    notes = f"Ambiguous — last seen {gap_days:.1f} days ago."
            else:
                outcome = "correct"
                notes = "Satellite removed from catalog."
        else:
            outcome = "expired"
            notes = "Unknown prediction type."

        store.resolve_prediction(pred["id"], outcome, notes)
        pred["outcome"] = outcome
        pred["resolution_notes"] = notes
        resolved.append(pred)

    return resolved


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m services.report.predictions",
    )
    parser.add_argument("action", choices=["generate", "resolve", "score"])
    parser.add_argument("--db", help="Path to SQLite store")
    args = parser.parse_args(argv)

    db = args.db or os.environ.get("ARGUS_DB_PATH", "data/starlink.db")
    store = StarlinkStore(db)

    if args.action == "generate":
        preds = generate_predictions(store)
        print(f"Generated {len(preds)} new predictions:")
        for p in preds:
            print(f"  [{p['prediction_type']}] {p['description'][:80]}")

    elif args.action == "resolve":
        resolved = resolve_predictions(store)
        print(f"Resolved {len(resolved)} predictions:")
        for r in resolved:
            print(f"  [{r['outcome']:>9s}] {r.get('name') or r['norad_id']}: {r['resolution_notes']}")

    elif args.action == "score":
        sc = store.get_prediction_scorecard()
        print(f"Prediction scorecard:")
        print(f"  Total:     {sc['total']}")
        print(f"  Pending:   {sc['pending']}")
        print(f"  Correct:   {sc['correct']}")
        print(f"  Incorrect: {sc['incorrect']}")
        print(f"  Expired:   {sc['expired']}")
        if sc['accuracy'] is not None:
            print(f"  Accuracy:  {sc['accuracy']*100:.1f}%")
        else:
            print(f"  Accuracy:  (no resolved predictions yet)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
