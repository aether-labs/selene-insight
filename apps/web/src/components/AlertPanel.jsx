import React from "react";

const TYPE_COLORS = {
  orbital_maneuver: "#ffaa00",
  sensor_anomaly: "#ff4444",
  coast_nominal: "#00ff88",
};

export default function AlertPanel({ alerts }) {
  const recent = alerts.slice(-5).reverse();

  return (
    <div style={styles.panel}>
      <h3 style={styles.heading}>
        SKEPTIC ALERTS
        {alerts.length > 0 && (
          <span style={styles.badge}>{alerts.length}</span>
        )}
      </h3>
      {recent.length === 0 ? (
        <p style={styles.noData}>No anomalies detected</p>
      ) : (
        recent.map((a, i) => (
          <div key={i} style={styles.alert}>
            <div style={styles.alertHeader}>
              <span
                style={{
                  ...styles.alertType,
                  color: TYPE_COLORS[a.alert_type] || "#888",
                }}
              >
                {a.alert_type?.toUpperCase().replace("_", " ")}
              </span>
              <span style={styles.alertMet}>{a.met}</span>
            </div>
            {a.details && (
              <p style={styles.alertDetails}>{a.details}</p>
            )}
            {a.deviation_pct != null && (
              <span style={styles.deviation}>
                {a.deviation_pct.toFixed(2)}% deviation
              </span>
            )}
          </div>
        ))
      )}
    </div>
  );
}

const styles = {
  panel: { marginBottom: 16 },
  heading: {
    fontSize: 12,
    letterSpacing: 2,
    color: "#ffaa00",
    marginBottom: 8,
    borderBottom: "1px solid rgba(255,170,0,0.2)",
    paddingBottom: 4,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  badge: {
    background: "#ffaa00",
    color: "#000",
    borderRadius: 10,
    padding: "1px 7px",
    fontSize: 10,
    fontWeight: 700,
  },
  noData: { fontSize: 12, color: "#555", fontStyle: "italic" },
  alert: {
    background: "rgba(255,255,255,0.03)",
    borderRadius: 4,
    padding: 8,
    marginBottom: 6,
    borderLeft: "2px solid rgba(255,170,0,0.4)",
  },
  alertHeader: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  alertType: { fontSize: 10, fontWeight: 700, letterSpacing: 1 },
  alertMet: { fontSize: 10, color: "#666" },
  alertDetails: { fontSize: 11, color: "#aaa", margin: 0, lineHeight: 1.4 },
  deviation: { fontSize: 10, color: "#ff6644", marginTop: 4, display: "block" },
};
