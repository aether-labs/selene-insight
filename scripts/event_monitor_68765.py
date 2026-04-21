"""Poll Space-Track for NORAD 68765 (AST BlueBird 7) state changes.
On new TLE, new TIP (Tracking and Impact Prediction) message, or decay
flag set, emit a ready-to-tweet update line and append to the alert log.

Four independent signals from Space-Track, any one of which can fire
first:
  - gp_history: a new TLE fit (apo / peri / INCL / a trend)
  - tip:        18 SDS reentry prediction (DECAY_EPOCH + window)
  - decay:      decay event recorded (fires BEFORE satcat sync; most
                reliable "it's down" signal available to us)
  - satcat.DECAY_DATE: official post-reentry confirmation (slowest;
                hours-to-days lag behind the decay class)

Trend classification uses all of (Δa, Δperi, Δapo, Δincl) because drag
only decreases a and never changes inclination; any incl change or any
a increase is a burn.

State is persisted in state.json so we only alert on true changes.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

NORAD = "68765"
NAME = "AST BLUEBIRD 7"
EPOCH_SECO1 = "2026-04-19T11:38:04"
APO_SECO1 = 493.610
PERI_SECO1 = 154.349
INCL_SECO1 = 36.1050
STATE_FILE = Path("/app/data/alerts/state.json")
LOG_FILE = Path("/app/data/alerts/alert.log")

R_EARTH = 6378.135  # km

USER = os.environ.get("SPACETRACK_USER")
PASS = os.environ.get("SPACETRACK_PASS")
if not USER or not PASS:
    sys.exit("SPACETRACK_USER/SPACETRACK_PASS not set")


def fetch() -> tuple[list[dict], dict, list[dict], list[dict]]:
    with httpx.Client(timeout=60, headers={"User-Agent": "argusorb/alert"}) as c:
        c.post(
            "https://www.space-track.org/ajaxauth/login",
            data={"identity": USER, "password": PASS},
        ).raise_for_status()
        hist = c.get(
            "https://www.space-track.org/basicspacedata/query/class/gp_history"
            f"/NORAD_CAT_ID/{NORAD}"
            "/orderby/EPOCH desc/format/json"
        ).json()
        satcat_rows = c.get(
            "https://www.space-track.org/basicspacedata/query/class/satcat"
            f"/NORAD_CAT_ID/{NORAD}/format/json"
        ).json()
        tip = c.get(
            "https://www.space-track.org/basicspacedata/query/class/tip"
            f"/NORAD_CAT_ID/{NORAD}"
            "/orderby/MSG_EPOCH desc/format/json"
        ).json()
        decay = c.get(
            "https://www.space-track.org/basicspacedata/query/class/decay"
            f"/NORAD_CAT_ID/{NORAD}"
            "/orderby/MSG_EPOCH desc/format/json"
        ).json()
    satcat = satcat_rows[0] if satcat_rows else {}
    return hist, satcat, tip, decay


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_epoch": None,
        "last_apoapsis": None,
        "last_periapsis": None,
        "last_inclination": None,
        "last_bstar": None,
        "decay_date": None,
        "last_tip_id": None,
        "last_decay_msg_epoch": None,
    }


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2))


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def parse_epoch(s: str) -> datetime:
    """Parse Space-Track datetime strings. Handles both ISO-strict formats
    (gp_history EPOCH is padded) and the loose format used in decay_msg
    and some catalog fields (e.g. '2026-04-20 0:00:00' with unpadded hour)."""
    s = s.replace("Z", "").strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Last resort: isoformat (handles timezone offsets etc.)
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def classify_trend(d_a: float, d_incl: float, d_peri: float, d_apo: float) -> str:
    """Physics-based classification of what a TLE change signals.

    Drag can only decrease semi-major axis and never changes inclination.
    So any |Δi| > 0.1° or Δa > 0.5 km proves a burn happened.
    """
    if abs(d_incl) > 0.1:
        return f"plane-change burn (Δi = {d_incl:+.2f}°)"
    if d_a > 0.5:
        return f"orbit-raising burn (Δa = {d_a:+.1f} km)"
    if d_peri > 0.5 and d_apo < -0.5:
        return "apoapsis burn (peri up, apo down)"
    if d_a < -0.5 and abs(d_apo) > abs(d_peri):
        return "drag decay (apoapsis erosion)"
    if d_a < -0.5:
        return f"orbit-lowering (Δa = {d_a:+.1f} km)"
    return "minor update (no significant change)"


def main() -> int:
    hist, satcat, tip_msgs, decay_msgs = fetch()
    state = load_state()

    changed = False

    # Signal 1: decay class record — fires BEFORE satcat.DECAY_DATE
    # populates, so it's our earliest reliable "decayed" signal.
    if decay_msgs:
        latest_decay = decay_msgs[0]
        decay_msg_epoch = latest_decay.get("MSG_EPOCH")
        if decay_msg_epoch != state.get("last_decay_msg_epoch"):
            decay_epoch = latest_decay.get("DECAY_EPOCH")
            msg_type = latest_decay.get("MSG_TYPE")
            log(f"STATE CHANGE: decay class record (MSG_EPOCH={decay_msg_epoch})")
            log(
                f"  DECAY_EPOCH={decay_epoch}  MSG_TYPE={msg_type}  "
                f"source={latest_decay.get('SOURCE')}"
            )
            log("TWEET:")
            log(
                f"  {NAME} (NORAD {NORAD}) reentered. Space-Track decay record "
                f"inserted {decay_msg_epoch}Z, DECAY_EPOCH {decay_epoch}Z. "
                f"From 154×494 km at SECO-1 to decay in "
                f"{_dt_delta(EPOCH_SECO1, decay_epoch)}."
            )
            state["last_decay_msg_epoch"] = decay_msg_epoch
            changed = True

    # Signal 2: official satcat.DECAY_DATE set (slowest; usually post-decay-class)
    sc_decay = satcat.get("DECAY_DATE")
    if sc_decay and sc_decay != state.get("decay_date"):
        log("STATE CHANGE: satcat.DECAY_DATE set")
        log(
            f"TWEET: {NAME} (NORAD {NORAD}) reentry confirmed in satcat on {sc_decay}. "
            f"From 154×494 km at SECO-1 (2026-04-19 11:38Z) to catalog decay in "
            f"{_dt_delta(EPOCH_SECO1, sc_decay)}."
        )
        state["decay_date"] = sc_decay
        changed = True

    # Signal 2: new TIP message (earliest leading indicator of reentry)
    if tip_msgs:
        latest_tip = tip_msgs[0]
        tip_id = latest_tip.get("ID") or latest_tip.get("MSG_EPOCH")
        if tip_id != state.get("last_tip_id"):
            decay_epoch = latest_tip.get("DECAY_EPOCH")
            window_min = int(latest_tip.get("WINDOW") or 0)
            lat = latest_tip.get("LAT")
            lon = latest_tip.get("LON")
            window_str = (
                f"±{window_min // 60}h" if window_min >= 120 else f"±{window_min}m"
            )
            log(f"STATE CHANGE: new TIP id={tip_id}")
            log(
                f"  predicted DECAY_EPOCH={decay_epoch}  window={window_str}  "
                f"impact lat={lat} lon={lon}"
            )
            log("TWEET:")
            log(
                f"  New 18 SDS TIP for {NAME} (NORAD {NORAD}): "
                f"predicted reentry {decay_epoch} UTC ({window_str}). "
                f"Impact point lat={lat} lon={lon}."
            )
            state["last_tip_id"] = tip_id
            changed = True

    # Signal 3: new TLEs in gp_history — process ALL unseen ones in chronological
    # order so we never skip intermediate burns (earlier bug: bootstrap jumped
    # straight to latest and missed T+1h intermediate TLE #2).
    if not hist:
        log("no gp_history — unexpected")
        return 1

    # Space-Track returned DESC; flip to chronological. Dedupe by EPOCH —
    # gp_history sometimes has duplicate entries for the same epoch (e.g.,
    # separate insertions for the same fit).
    seen_epochs: set[str] = set()
    hist_asc: list[dict] = []
    for r in sorted(hist, key=lambda x: x["EPOCH"]):
        if r["EPOCH"] in seen_epochs:
            continue
        seen_epochs.add(r["EPOCH"])
        hist_asc.append(r)

    last_seen_epoch = state.get("last_epoch")
    new_tles = [r for r in hist_asc if last_seen_epoch is None or r["EPOCH"] > last_seen_epoch]

    if not new_tles:
        if changed:
            save_state(state)
        else:
            log(
                f"no change; latest epoch still {hist_asc[-1]['EPOCH']}; "
                f"tip_msgs={len(tip_msgs)}, decay_msgs={len(decay_msgs)}, "
                f"gp_count={len(hist)}"
            )
        return 0

    # Previous state baseline (used for the first iteration)
    prev_apo = state.get("last_apoapsis") or APO_SECO1
    prev_peri = state.get("last_periapsis") or PERI_SECO1
    prev_incl = state.get("last_inclination") or INCL_SECO1

    log(
        f"processing {len(new_tles)} new TLE(s) "
        f"(gp_count={len(hist)}, tip_msgs={len(tip_msgs)}, "
        f"decay_msgs={len(decay_msgs)})"
    )

    for r in new_tles:
        epoch = r["EPOCH"]
        apo = float(r["APOAPSIS"])
        peri = float(r["PERIAPSIS"])
        incl = float(r["INCLINATION"])
        bstar = float(r["BSTAR"])
        mm = float(r["MEAN_MOTION"])
        a = (apo + peri) / 2 + R_EARTH
        prev_a = (prev_apo + prev_peri) / 2 + R_EARTH

        age_h = (datetime.now(timezone.utc) - parse_epoch(epoch)).total_seconds() / 3600
        since_seco1_h = (
            parse_epoch(epoch) - parse_epoch(EPOCH_SECO1)
        ).total_seconds() / 3600

        d_apo = apo - prev_apo
        d_peri = peri - prev_peri
        d_incl = incl - prev_incl
        d_a = a - prev_a

        c_apo = apo - APO_SECO1
        c_peri = peri - PERI_SECO1
        c_incl = incl - INCL_SECO1

        trend = classify_trend(d_a, d_incl, d_peri, d_apo)

        log(
            f"NEW TLE: epoch={epoch} ({age_h:.1f}h old, {since_seco1_h:.1f}h since SECO-1)"
        )
        log(
            f"  apo={apo:.1f} km (Δstep {d_apo:+.1f}, ΔSECO1 {c_apo:+.1f}), "
            f"peri={peri:.1f} km (Δstep {d_peri:+.1f}, ΔSECO1 {c_peri:+.1f}), "
            f"incl={incl:.3f}° (Δstep {d_incl:+.3f}, ΔSECO1 {c_incl:+.3f}), "
            f"bstar={bstar:+.4e}, a={a:.1f} km (Δstep {d_a:+.1f})"
        )
        log(f"  TREND: {trend}")

        if abs(d_incl) > 0.1:
            tweet = (
                f"New TLE for {NAME} at T+{since_seco1_h:.1f}h: "
                f"incl {prev_incl:.1f}° → {incl:.1f}° (Δ {d_incl:+.1f}°), "
                f"{peri:.0f} × {apo:.0f} km. "
                f"Plane change of this magnitude = active burn."
            )
        elif d_a > 0.5:
            tweet = (
                f"New TLE for {NAME} at T+{since_seco1_h:.1f}h: "
                f"apo {apo:.0f} km (Δ {d_apo:+.0f}), peri {peri:.0f} km (Δ {d_peri:+.0f}), "
                f"incl {incl:.1f}°. Semi-major axis up {d_a:+.1f} km — orbit-raising burn."
            )
        elif d_peri > 0.5 and d_apo < -0.5:
            tweet = (
                f"New TLE for {NAME} at T+{since_seco1_h:.1f}h: "
                f"perigee raised {prev_peri:.0f} → {peri:.0f} km (+{d_peri:.0f}), "
                f"apo {prev_apo:.0f} → {apo:.0f} km. Apogee burn signature."
            )
        elif d_a < -0.5:
            tweet = (
                f"New TLE for {NAME} at T+{since_seco1_h:.1f}h: "
                f"apo {apo:.0f} km (Δ {d_apo:+.0f}), peri {peri:.0f} km (Δ {d_peri:+.0f}), "
                f"incl {incl:.1f}°. a lost {-d_a:.1f} km — drag or retrograde burn."
            )
        else:
            tweet = (
                f"New TLE for {NAME} at T+{since_seco1_h:.1f}h: "
                f"apo {apo:.0f} km, peri {peri:.0f} km, incl {incl:.1f}°, B* {bstar:+.2e}. "
                f"No significant change from last."
            )

        log("TWEET:")
        log(f"  {tweet}")

        # Update baseline for next iteration in this run
        prev_apo, prev_peri, prev_incl = apo, peri, incl

    # Persist final state after processing all new TLEs
    final = new_tles[-1]
    state["last_epoch"] = final["EPOCH"]
    state["last_apoapsis"] = float(final["APOAPSIS"])
    state["last_periapsis"] = float(final["PERIAPSIS"])
    state["last_inclination"] = float(final["INCLINATION"])
    state["last_bstar"] = float(final["BSTAR"])
    save_state(state)
    return 0


def _dt_delta(t0_str: str, t1_str: str) -> str:
    t0 = parse_epoch(t0_str)
    try:
        t1 = parse_epoch(t1_str)
    except ValueError:
        t1 = datetime.fromisoformat(t1_str).replace(tzinfo=timezone.utc)
    h = (t1 - t0).total_seconds() / 3600
    if h < 48:
        return f"{h:.1f}h"
    return f"{h / 24:.1f} days"


if __name__ == "__main__":
    sys.exit(main())
