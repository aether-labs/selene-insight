import React, { useMemo, useRef, useEffect } from "react";
import {
  Viewer,
  Entity,
  PolylineGraphics,
  PointGraphics,
  LabelGraphics,
  EllipsoidGraphics,
} from "resium";
import {
  Cartesian3,
  Color,
  LabelStyle,
  VerticalOrigin,
  HorizontalOrigin,
  NearFarScalar,
} from "cesium";
import { telemetryToPosition, moonToPosition } from "../lib/orbit.js";
import {
  generateReferenceTrajectory,
  estimateLaunchTime,
} from "../lib/referenceTrajectory.js";

// Moon radius in meters
const MOON_RADIUS_M = 1.737e6;

function trajectoryPositions(telemetry) {
  return telemetry
    .map((p) => telemetryToPosition(p, Cartesian3))
    .filter(Boolean);
}

export default function CesiumView({ telemetry, latest }) {
  const viewerRef = useRef(null);
  const positions = useMemo(() => trajectoryPositions(telemetry), [telemetry]);
  const currentPos = useMemo(
    () => telemetryToPosition(latest, Cartesian3),
    [latest],
  );

  const moonPos = useMemo(() => {
    const ts = latest?.timestamp || Date.now() / 1000;
    return moonToPosition(ts, Cartesian3);
  }, [latest]);

  const refTrajectory = useMemo(() => {
    const launchTime = estimateLaunchTime(latest);
    if (!launchTime) return [];
    return generateReferenceTrajectory(launchTime, Cartesian3);
  }, [latest?.met]);

  // Fly camera to show full Earth-Moon system on first data
  const hasFlewRef = useRef(false);
  useEffect(() => {
    if (!hasFlewRef.current && currentPos && viewerRef.current?.cesiumElement) {
      const viewer = viewerRef.current.cesiumElement;
      viewer.camera.flyTo({
        destination: Cartesian3.fromDegrees(0, 0, 500_000_000),
        duration: 2,
      });
      hasFlewRef.current = true;
    }
  }, [currentPos]);

  return (
    <Viewer
      ref={viewerRef}
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
      {/* Reference trajectory (predicted full orbit) */}
      {refTrajectory.length > 1 && (
        <Entity name="Predicted Trajectory">
          <PolylineGraphics
            positions={refTrajectory}
            width={1.5}
            material={Color.fromCssColorString("#ffffff").withAlpha(0.2)}
          />
        </Entity>
      )}

      {/* Actual telemetry path */}
      {positions.length > 1 && (
        <Entity name="Actual Path">
          <PolylineGraphics
            positions={positions}
            width={3}
            material={Color.fromCssColorString("#00ccff").withAlpha(0.8)}
          />
        </Entity>
      )}

      {/* Orion — outer glow */}
      {currentPos && (
        <Entity position={currentPos}>
          <PointGraphics
            pixelSize={24}
            color={Color.fromCssColorString("#00ccff").withAlpha(0.3)}
          />
        </Entity>
      )}

      {/* Orion — core dot + label */}
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
          <LabelGraphics
            text="ORION"
            font="bold 14px monospace"
            fillColor={Color.CYAN}
            style={LabelStyle.FILL_AND_OUTLINE}
            outlineColor={Color.BLACK}
            outlineWidth={2}
            verticalOrigin={VerticalOrigin.BOTTOM}
            pixelOffset={{ x: 0, y: -18 }}
            scaleByDistance={new NearFarScalar(1e6, 1.0, 5e8, 0.4)}
          />
        </Entity>
      )}

      {/* Moon — sphere + label */}
      <Entity name="Moon" position={moonPos}>
        <EllipsoidGraphics
          radii={new Cartesian3(MOON_RADIUS_M, MOON_RADIUS_M, MOON_RADIUS_M)}
          material={Color.fromCssColorString("#cccccc").withAlpha(0.9)}
        />
        <LabelGraphics
          text="MOON"
          font="bold 14px monospace"
          fillColor={Color.fromCssColorString("#cccccc")}
          style={LabelStyle.FILL_AND_OUTLINE}
          outlineColor={Color.BLACK}
          outlineWidth={2}
          verticalOrigin={VerticalOrigin.BOTTOM}
          pixelOffset={{ x: 0, y: -20 }}
          scaleByDistance={new NearFarScalar(1e6, 1.0, 5e8, 0.4)}
        />
      </Entity>

      {/* Earth label */}
      <Entity position={Cartesian3.fromDegrees(0, 90, 0)}>
        <LabelGraphics
          text="EARTH"
          font="bold 14px monospace"
          fillColor={Color.fromCssColorString("#4488ff")}
          style={LabelStyle.FILL_AND_OUTLINE}
          outlineColor={Color.BLACK}
          outlineWidth={2}
          verticalOrigin={VerticalOrigin.BOTTOM}
          pixelOffset={{ x: 0, y: -10 }}
          scaleByDistance={new NearFarScalar(1e6, 1.0, 5e8, 0.4)}
        />
      </Entity>
    </Viewer>
  );
}
