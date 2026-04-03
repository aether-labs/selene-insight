import React from "react";

export default function TelemetryPanel({ latest, count }) {
  if (!latest) {
    return (
      <div style={styles.panel}>
        <h3 style={styles.heading}>TELEMETRY</h3>
        <p style={styles.noData}>Waiting for data...</p>
      </div>
    );
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>TELEMETRY</h3>
      <Row label="MET" value={latest.met} />
      <Row label="PHASE" value={latest.phase} />
      <Row label="VELOCITY" value={`${latest.velocity_kms?.toFixed(3)} km/s`} />
      <Row label="EARTH" value={`${latest.earth_dist_km?.toLocaleString()} km`} />
      <Row label="MOON" value={`${latest.moon_dist_km?.toLocaleString()} km`} />
      <div style={styles.meta}>{count} readings buffered</div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={styles.row}>
      <span style={styles.label}>{label}</span>
      <span style={styles.value}>{value || "—"}</span>
    </div>
  );
}

const styles = {
  panel: { marginBottom: 16 },
  heading: {
    fontSize: 12,
    letterSpacing: 2,
    color: "#00ccff",
    marginBottom: 8,
    borderBottom: "1px solid rgba(0,204,255,0.2)",
    paddingBottom: 4,
  },
  row: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    padding: "3px 0",
  },
  label: { color: "#888", fontWeight: 500 },
  value: { color: "#e0e0e0", fontFamily: "monospace" },
  noData: { fontSize: 12, color: "#555", fontStyle: "italic" },
  meta: { fontSize: 10, color: "#555", marginTop: 6 },
};
