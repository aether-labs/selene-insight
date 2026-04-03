import React, { useState, useEffect, useRef, useCallback } from "react";
import CesiumView from "./components/CesiumView.jsx";
import TelemetryPanel from "./components/TelemetryPanel.jsx";
import AlertPanel from "./components/AlertPanel.jsx";
import TimeSlider from "./components/TimeSlider.jsx";
import { useTelemetryWs } from "./hooks/useTelemetryWs.js";

export default function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [liveMode, setLiveMode] = useState(true);
  const [sliderTime, setSliderTime] = useState(null);

  // WebSocket for live data
  const onTelemetry = useCallback((point) => {
    setTelemetry((prev) => [...prev.slice(-999), point]);
  }, []);

  const onAlert = useCallback((alert) => {
    setAlerts((prev) => [...prev.slice(-99), alert]);
  }, []);

  useTelemetryWs({ onTelemetry, onAlert, enabled: liveMode });

  // Fetch initial data on mount
  useEffect(() => {
    fetch("/api/telemetry/latest?n=100")
      .then((r) => r.json())
      .then((d) => setTelemetry(d.data || []))
      .catch(() => {});
    fetch("/api/alerts/latest?n=50")
      .then((r) => r.json())
      .then((d) => setAlerts(d.data || []))
      .catch(() => {});
  }, []);

  // Time range query for rewind
  const handleSliderChange = useCallback(async (start, end) => {
    setLiveMode(false);
    setSliderTime({ start, end });
    try {
      const r = await fetch(
        `/api/telemetry/range?start=${start}&end=${end}&limit=2000`
      );
      const d = await r.json();
      setTelemetry(d.data || []);
    } catch {
      /* offline fallback */
    }
  }, []);

  const handleGoLive = useCallback(() => {
    setLiveMode(true);
    setSliderTime(null);
  }, []);

  const latest = telemetry[telemetry.length - 1] || null;

  return (
    <div style={styles.container}>
      {/* 3D Globe */}
      <div style={styles.globe}>
        <CesiumView telemetry={telemetry} latest={latest} />
      </div>

      {/* Overlay panels */}
      <div style={styles.sidebar}>
        <h1 style={styles.title}>SELENE-INSIGHT</h1>
        <p style={styles.subtitle}>Artemis II Digital Twin</p>
        <TelemetryPanel latest={latest} count={telemetry.length} />
        <AlertPanel alerts={alerts} />
      </div>

      {/* Time slider */}
      <div style={styles.sliderBar}>
        <TimeSlider
          liveMode={liveMode}
          onRangeChange={handleSliderChange}
          onGoLive={handleGoLive}
        />
      </div>

      {/* Connection indicator */}
      <div style={styles.indicator}>
        <span
          style={{
            ...styles.dot,
            background: liveMode ? "#00ff88" : "#ffaa00",
          }}
        />
        {liveMode ? "LIVE" : "REWIND"}
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: "100%",
    height: "100%",
    position: "relative",
  },
  globe: {
    width: "100%",
    height: "100%",
    position: "absolute",
    top: 0,
    left: 0,
  },
  sidebar: {
    position: "absolute",
    top: 16,
    left: 16,
    width: 340,
    background: "rgba(10, 10, 20, 0.85)",
    borderRadius: 8,
    padding: 16,
    backdropFilter: "blur(12px)",
    border: "1px solid rgba(255,255,255,0.08)",
    maxHeight: "calc(100vh - 100px)",
    overflowY: "auto",
    zIndex: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: 3,
    color: "#00ccff",
    marginBottom: 2,
  },
  subtitle: {
    fontSize: 11,
    color: "#888",
    marginBottom: 16,
    letterSpacing: 1,
  },
  sliderBar: {
    position: "absolute",
    bottom: 16,
    left: "50%",
    transform: "translateX(-50%)",
    width: "60%",
    minWidth: 400,
    zIndex: 10,
  },
  indicator: {
    position: "absolute",
    top: 16,
    right: 16,
    background: "rgba(10, 10, 20, 0.85)",
    borderRadius: 6,
    padding: "8px 16px",
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: 2,
    display: "flex",
    alignItems: "center",
    gap: 8,
    backdropFilter: "blur(12px)",
    border: "1px solid rgba(255,255,255,0.08)",
    zIndex: 10,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    display: "inline-block",
  },
};
