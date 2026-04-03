import React, { useMemo } from "react";
import {
  Viewer,
  Entity,
  PolylineGraphics,
  PointGraphics,
} from "resium";
import { Cartesian3, Color } from "cesium";
import { telemetryToPosition, moonToPosition } from "../lib/orbit.js";

function trajectoryPositions(telemetry) {
  return telemetry
    .map((p) => telemetryToPosition(p, Cartesian3))
    .filter(Boolean);
}

export default function CesiumView({ telemetry, latest }) {
  const positions = useMemo(() => trajectoryPositions(telemetry), [telemetry]);
  const currentPos = useMemo(
    () => telemetryToPosition(latest, Cartesian3),
    [latest],
  );

  // Moon position at latest timestamp (or now)
  const moonPos = useMemo(() => {
    const ts = latest?.timestamp || Date.now() / 1000;
    return moonToPosition(ts, Cartesian3);
  }, [latest]);

  return (
    <Viewer
      full
      timeline={false}
      animation={false}
      homeButton={false}
      geocoder={false}
      sceneModePicker={false}
      baseLayerPicker={false}
      navigationHelpButton={false}
      style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
    >
      {/* Orion trajectory path */}
      {positions.length > 1 && (
        <Entity>
          <PolylineGraphics
            positions={positions}
            width={2}
            material={Color.fromCssColorString("#00ccff").withAlpha(0.7)}
          />
        </Entity>
      )}

      {/* Current Orion position */}
      {currentPos && (
        <Entity
          name="Orion"
          position={currentPos}
          description={
            latest
              ? `MET: ${latest.met}<br/>` +
                `Velocity: ${latest.velocity_kms?.toFixed(3)} km/s<br/>` +
                `Earth: ${latest.earth_dist_km?.toFixed(0)} km<br/>` +
                `Moon: ${latest.moon_dist_km?.toFixed(0)} km<br/>` +
                `Phase: ${latest.phase}`
              : ""
          }
        >
          <PointGraphics pixelSize={10} color={Color.CYAN} />
        </Entity>
      )}

      {/* Moon */}
      <Entity name="Moon" position={moonPos}>
        <PointGraphics pixelSize={14} color={Color.LIGHTGRAY} />
      </Entity>
    </Viewer>
  );
}
