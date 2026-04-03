import React, { useState, useCallback } from "react";

/**
 * Time slider for rewinding through telemetry history.
 * Queries the API for a time range when the user drags.
 */
export default function TimeSlider({ liveMode, onRangeChange, onGoLive }) {
  // Slider represents minutes-ago (0 = now, max = 60 min ago)
  const [value, setValue] = useState(0);
  const maxMinutes = 60;

  const handleChange = useCallback(
    (e) => {
      const minutesAgo = Number(e.target.value);
      setValue(minutesAgo);
      if (minutesAgo === 0) {
        onGoLive();
      } else {
        const now = Date.now() / 1000;
        const windowSec = 60; // 1-minute window around the selected time
        const center = now - minutesAgo * 60;
        onRangeChange(center - windowSec / 2, center + windowSec / 2);
      }
    },
    [onRangeChange, onGoLive]
  );

  const handleGoLive = useCallback(() => {
    setValue(0);
    onGoLive();
  }, [onGoLive]);

  return (
    <div style={styles.container}>
      <div style={styles.bar}>
        <span style={styles.label}>
          {value === 0 ? "NOW" : `${value}m ago`}
        </span>
        <input
          type="range"
          min={0}
          max={maxMinutes}
          value={value}
          onChange={handleChange}
          style={styles.slider}
        />
        <button
          onClick={handleGoLive}
          style={{
            ...styles.liveBtn,
            opacity: liveMode ? 0.4 : 1,
          }}
          disabled={liveMode}
        >
          GO LIVE
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: "rgba(10, 10, 20, 0.85)",
    borderRadius: 8,
    padding: "10px 16px",
    backdropFilter: "blur(12px)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  bar: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  label: {
    fontSize: 11,
    fontWeight: 600,
    color: "#00ccff",
    minWidth: 60,
    letterSpacing: 1,
  },
  slider: {
    flex: 1,
    height: 4,
    appearance: "auto",
    accentColor: "#00ccff",
    cursor: "pointer",
  },
  liveBtn: {
    background: "#00ff88",
    color: "#000",
    border: "none",
    borderRadius: 4,
    padding: "4px 12px",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1,
    cursor: "pointer",
  },
};
