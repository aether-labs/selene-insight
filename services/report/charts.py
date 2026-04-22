"""Chart generator for the weekly report.

Produces publication-ready PNG charts from ArgusOrb data. Each chart
is designed for Substack embedding (1200×630 px, dark theme, high DPI).

Charts:
  1. Shell population bar chart (where are the satellites?)
  2. Anomaly breakdown donut (what happened this week?)
  3. Satellite event timeline (eccentricity/B* for a specific NORAD ID)
  4. B* fleet distribution histogram (propulsion behavior)

Usage:
    from services.report.charts import generate_all_charts
    paths = generate_all_charts(store, start_ts, end_ts, output_dir)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Dark theme matching argusorb.io ──
DARK_BG = "#0a0e1a"
PANEL_BG = "#0f1424"
TEXT_COLOR = "#bbccdd"
ACCENT = "#00ccff"
ORANGE = "#ffaa00"
RED = "#ff4444"
GREEN = "#00ff88"
GRID_COLOR = "#1a2040"

CHART_WIDTH = 12       # inches at 100 dpi = 1200 px
CHART_HEIGHT = 6.3     # → 630 px
DPI = 100


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "font.size": 11,
    })


def chart_shell_population(store, output_dir: Path) -> Path:
    """Horizontal bar chart: satellite count per orbital shell."""
    if not HAS_MPL:
        return None

    _apply_dark_theme()
    shells = store.count_fresh_by_shell(
        as_of_ts=__import__("time").time(), freshness_s=86400
    )

    # Group small shells into "other"
    MIN_POP = 50
    grouped = {}
    other = 0
    for km, n in sorted(shells.items()):
        if km <= 0:
            grouped["decayed"] = grouped.get("decayed", 0) + n
        elif n >= MIN_POP:
            grouped[f"{int(km)} km"] = n
        else:
            other += n
    if other > 0:
        grouped["other"] = other

    labels = list(grouped.keys())
    values = list(grouped.values())
    colors = [ACCENT if v > 1000 else "#4488ff" if v > 200 else "#667788" for v in values]

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Satellites")
    ax.set_title("STARLINK CONSTELLATION — SHELL POPULATION", fontsize=14,
                 color=ACCENT, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(values) * 1.15)

    path = output_dir / "shell_population.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def chart_anomaly_breakdown(store, start_ts: float, end_ts: float,
                            output_dir: Path) -> Path:
    """Donut chart: anomaly labels by cause this week."""
    if not HAS_MPL:
        return None

    _apply_dark_theme()
    anomalies = store.get_anomalies_in_window(start_ts, min(end_ts, __import__("time").time()),
                                               classified_by="rule_v1")
    if not anomalies:
        return None

    from collections import Counter
    causes = Counter(a.get("cause", "unknown") for a in anomalies)

    labels = list(causes.keys())
    sizes = list(causes.values())
    color_map = {
        "maneuver_candidate": ACCENT,
        "atmospheric_anomaly": ORANGE,
        "natural_decay": "#ff6644",
        "reentry": RED,
    }
    colors = [color_map.get(l, "#667788") for l in labels]
    display_labels = [l.replace("_", " ").title() for l in labels]

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=display_labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.82, labeldistance=1.08,
        wedgeprops=dict(width=0.35, edgecolor=DARK_BG, linewidth=2),
    )
    for t in texts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(11)
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10)
        t.set_fontweight("bold")

    ax.set_title(f"FLAGGED EVENTS — rule_v1 ({sum(sizes):,} total)",
                 fontsize=14, color=ACCENT, fontweight="bold", pad=15)

    path = output_dir / "anomaly_breakdown.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def chart_satellite_timeline(store, norad_id: int, output_dir: Path,
                             title: str = None) -> Path:
    """Dual-axis timeline: eccentricity + B* for a specific satellite."""
    if not HAS_MPL:
        return None

    _apply_dark_theme()
    history = store.get_satellite_history(norad_id, limit=500)
    if len(history) < 5:
        return None

    history.reverse()  # oldest first
    sat = store.get_satellite(norad_id)
    name = sat.get("name", f"NORAD {norad_id}") if sat else f"NORAD {norad_id}"

    epochs = []
    eccs = []
    bstars = []
    for h in history:
        jd = h.get("epoch_jd", 0)
        if jd > 0:
            # Convert JD to datetime (approximate)
            dt = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc) + \
                 __import__("datetime").timedelta(days=jd - 2451545.0)
            epochs.append(dt)
            eccs.append(h.get("eccentricity") or 0)
            bstars.append(h.get("bstar"))

    fig, ax1 = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    # Eccentricity (left axis)
    color_ecc = ACCENT
    ax1.plot(epochs, eccs, color=color_ecc, linewidth=1.5, alpha=0.9, label="Eccentricity")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Eccentricity", color=color_ecc)
    ax1.tick_params(axis="y", labelcolor=color_ecc)
    ax1.grid(alpha=0.2)

    # B* (right axis)
    valid_bstars = [(e, b) for e, b in zip(epochs, bstars) if b is not None]
    if valid_bstars:
        b_epochs, b_vals = zip(*valid_bstars)
        ax2 = ax1.twinx()
        color_bstar = ORANGE
        ax2.plot(b_epochs, b_vals, color=color_bstar, linewidth=1.0, alpha=0.7, label="B*")
        ax2.set_ylabel("B* drag coefficient", color=color_bstar)
        ax2.tick_params(axis="y", labelcolor=color_bstar)
        ax2.axhline(y=0, color=color_bstar, linewidth=0.5, alpha=0.3, linestyle="--")

    title_text = title or f"{name} — ORBITAL ELEMENTS TIMELINE"
    ax1.set_title(title_text, fontsize=14, color=ACCENT, fontweight="bold", pad=15)

    fig.autofmt_xdate()

    path = output_dir / f"timeline_{norad_id}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def chart_bstar_distribution(store, output_dir: Path) -> Path:
    """Histogram: B* positive/negative distribution across the fleet."""
    if not HAS_MPL:
        return None

    _apply_dark_theme()
    tles = store.get_latest_tles()
    bstars = [t.get("bstar") for t in tles if t.get("bstar") is not None]
    if not bstars:
        return None

    bstars = np.array(bstars)
    pos = bstars[bstars > 0]
    neg = bstars[bstars < 0]

    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))

    bins = np.linspace(-0.06, 0.06, 80)
    ax.hist(neg, bins=bins, color=RED, alpha=0.7, label=f"Negative ({len(neg):,}) — thrust")
    ax.hist(pos, bins=bins, color=GREEN, alpha=0.7, label=f"Positive ({len(pos):,}) — coast")

    ax.axvline(x=0, color="white", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_xlabel("B* drag coefficient")
    ax.set_ylabel("Satellites")
    ax.set_title("STARLINK FLEET B* DISTRIBUTION — PROPULSION BEHAVIOR",
                 fontsize=14, color=ACCENT, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.2)

    pct_neg = len(neg) / len(bstars) * 100
    pct_pos = len(pos) / len(bstars) * 100
    ax.text(0.02, 0.95, f"{pct_neg:.0f}% thrust / {pct_pos:.0f}% coast",
            transform=ax.transAxes, fontsize=12, color="white", alpha=0.8,
            verticalalignment="top")

    path = output_dir / "bstar_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def generate_all_charts(
    store,
    start_ts: float,
    end_ts: float,
    output_dir: Path,
    highlight_norad_ids: list[int] | None = None,
) -> list[Path]:
    """Generate all charts for the weekly report. Returns list of paths."""
    if not HAS_MPL:
        print("[charts] matplotlib not available, skipping charts")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    print("  [charts] shell population...")
    p = chart_shell_population(store, output_dir)
    if p:
        paths.append(p)

    print("  [charts] anomaly breakdown...")
    p = chart_anomaly_breakdown(store, start_ts, end_ts, output_dir)
    if p:
        paths.append(p)

    print("  [charts] B* distribution...")
    p = chart_bstar_distribution(store, output_dir)
    if p:
        paths.append(p)

    # Satellite timelines
    # Use highlighted IDs if provided, otherwise pick satellites with most history
    timeline_ids = []
    if highlight_norad_ids:
        timeline_ids = highlight_norad_ids[:3]
    if not timeline_ids:
        # Pick 2 satellites with the most TLE records
        all_tles = store.get_latest_tles()
        candidates = []
        for sat in all_tles[:100]:  # check first 100
            h = store.get_satellite_history(sat["norad_id"], limit=500)
            if len(h) >= 10:
                candidates.append((sat["norad_id"], len(h), sat.get("name", "")))
        candidates.sort(key=lambda x: -x[1])
        timeline_ids = [c[0] for c in candidates[:2]]

    for nid in timeline_ids:
        print(f"  [charts] timeline NORAD {nid}...")
        p = chart_satellite_timeline(store, nid, output_dir)
        if p:
            paths.append(p)

    print(f"  [charts] generated {len(paths)} charts")
    return paths
