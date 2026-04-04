/**
 * Selene-Insight — Full-screen CesiumJS dashboard with time animation.
 */

import {
  Viewer,
  Cartesian3,
  Cartesian2,
  Color,
  Ion,
  LabelStyle,
  VerticalOrigin,
  NearFarScalar,
  PolylineDashMaterialProperty,
  SceneMode,
  JulianDate,
  ClockRange,
  ClockStep,
  SampledPositionProperty,
  LagrangePolynomialApproximation,
  TimeIntervalCollection,
  TimeInterval,
  PathGraphics,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { moonToPosition, eciToCartesian3 } from "./lib/orbit.js";
import { generateMoonOrbit } from "./lib/referenceTrajectory.js";

// ── Config ──
Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

const MOON_RADIUS_M = 1.737e6;
const LABEL_FONT = "bold 22px monospace";

// ── State ──
let alertBuffer = [];

// ── CesiumJS viewer ──
const viewer = new Viewer("cesium-container", {
  timeline: true,       // enable for animation scrubbing
  animation: true,      // play/pause/speed controls
  homeButton: false,
  geocoder: false,
  sceneModePicker: false,
  baseLayerPicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  infoBox: true,
  selectionIndicator: false,
  sceneMode: SceneMode.SCENE3D,
  skyBox: false,
  skyAtmosphere: false,
});
viewer.scene.backgroundColor = Color.BLACK;

// Camera
viewer.camera.setView({
  destination: Cartesian3.fromDegrees(20, 30, 800_000_000),
});

// ── Static Entities ──

const nowTs = Date.now() / 1000;

// Moon orbit ring
viewer.entities.add({
  name: "Moon Orbit",
  polyline: {
    positions: generateMoonOrbit(nowTs, Cartesian3),
    width: 1,
    material: Color.fromCssColorString("#444444").withAlpha(0.35),
  },
});

// Moon
const moonEntity = viewer.entities.add({
  name: "Moon",
  position: moonToPosition(nowTs, Cartesian3),
  ellipsoid: {
    radii: new Cartesian3(MOON_RADIUS_M, MOON_RADIUS_M, MOON_RADIUS_M),
    material: Color.fromCssColorString("#cccccc").withAlpha(0.9),
  },
  label: {
    text: "MOON",
    font: LABEL_FONT,
    fillColor: Color.fromCssColorString("#cccccc"),
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -28),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
});

// Earth — slight exaggeration (1.5x) for visibility
const EARTH_VIS_RADIUS = 6371000 * 1.5;
viewer.entities.add({
  name: "Earth",
  position: Cartesian3.fromDegrees(0, 0, 0),
  ellipsoid: {
    radii: new Cartesian3(EARTH_VIS_RADIUS, EARTH_VIS_RADIUS, EARTH_VIS_RADIUS),
    material: Color.fromCssColorString("#2244aa").withAlpha(0.8),
  },
  label: {
    text: "EARTH",
    font: LABEL_FONT,
    fillColor: Color.fromCssColorString("#4488ff"),
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -18),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
});

// Orion entity — position set by SampledPositionProperty after history loads
const orionPosition = new SampledPositionProperty();
orionPosition.setInterpolationOptions({
  interpolationDegree: 3,
  interpolationAlgorithm: LagrangePolynomialApproximation,
});

const orionEntity = viewer.entities.add({
  name: "Orion",
  position: orionPosition,
  point: { pixelSize: 14, color: Color.CYAN, outlineColor: Color.WHITE, outlineWidth: 2 },
  label: {
    text: "ORION",
    font: LABEL_FONT,
    fillColor: Color.CYAN,
    style: LabelStyle.FILL_AND_OUTLINE,
    outlineColor: Color.BLACK,
    outlineWidth: 4,
    verticalOrigin: VerticalOrigin.BOTTOM,
    pixelOffset: new Cartesian2(0, -28),
    scaleByDistance: new NearFarScalar(5e5, 1.6, 1e9, 0.6),
  },
  // Trail: shows path Orion has already flown
  path: new PathGraphics({
    leadTime: 0,             // no future path (only past)
    trailTime: 86400 * 10,   // show up to 10 days of trail
    width: 4,
    material: Color.fromCssColorString("#00ccff").withAlpha(0.85),
  }),
});

// ── DOM refs ──
const dom = {
  met: document.getElementById("t-met"),
  phase: document.getElementById("t-phase"),
  velocity: document.getElementById("t-velocity"),
  earth: document.getElementById("t-earth"),
  moon: document.getElementById("t-moon"),
  source: document.getElementById("t-source"),
  count: document.getElementById("t-count"),
  grade: document.getElementById("v-grade"),
  confidence: document.getElementById("v-confidence"),
  vVel: document.getElementById("v-vel"),
  vEarth: document.getElementById("v-earth"),
  vMoon: document.getElementById("v-moon"),
  vCount: document.getElementById("v-count"),
  alertList: document.getElementById("alert-list"),
  connStatus: document.getElementById("connection-status"),
};

// ── Load Horizons history and set up time animation ──

fetch("/api/telemetry/history")
  .then((r) => r.json())
  .then((d) => {
    if (!d.data || !d.data.length) return;

    const points = d.data.filter((p) => p.pos_km);
    if (points.length < 2) return;

    // Add samples to Orion's position property
    for (const p of points) {
      const jd = JulianDate.fromDate(new Date(p.timestamp * 1000));
      const pos = eciToCartesian3(p.pos_km, Cartesian3);
      orionPosition.addSample(jd, pos);
    }

    // Set clock to mission time range
    const startJd = JulianDate.fromDate(new Date(points[0].timestamp * 1000));
    const stopJd = JulianDate.fromDate(new Date(points[points.length - 1].timestamp * 1000));
    const nowJd = JulianDate.fromDate(new Date());

    viewer.clock.startTime = startJd.clone();
    viewer.clock.stopTime = stopJd.clone();
    viewer.clock.currentTime = nowJd.clone();
    viewer.clock.clockRange = ClockRange.CLAMPED;
    viewer.clock.clockStep = ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    viewer.clock.multiplier = 1; // real-time

    viewer.timeline.zoomTo(startJd, stopJd);

    console.log(`[HISTORY] Loaded ${points.length} Horizons waypoints`);

    // Update telemetry panel with latest
    const latest = points[points.length - 1];
    updateTelemetryDOM(latest);
  })
  .catch((e) => console.warn("[HISTORY] Failed:", e));

function updateTelemetryDOM(point) {
  dom.met.textContent = point.met || "--";
  dom.phase.textContent = point.phase || "--";
  dom.velocity.textContent = point.velocity_kms ? `${point.velocity_kms.toFixed(3)} km/s` : "--";
  dom.earth.textContent = point.earth_dist_km ? `${Math.round(point.earth_dist_km).toLocaleString()} km` : "--";
  dom.moon.textContent = point.moon_dist_km ? `${Math.round(point.moon_dist_km).toLocaleString()} km` : "--";
  dom.source.textContent = point.source || "jpl_horizons";
}

// ── Fetch alerts and validation ──

fetch("/api/alerts/latest?n=10")
  .then((r) => r.json())
  .then((d) => { if (d.data) d.data.reverse().forEach(addAlert); })
  .catch(() => {});

fetch("/api/validation/latest")
  .then((r) => r.json())
  .then((d) => {
    if (d.recent && d.recent.length) updateValidation(d.recent[d.recent.length - 1]);
    if (d.stats) dom.vCount.textContent = d.stats.total_validations || 0;
  })
  .catch(() => {});

function updateValidation(data) {
  const grade = data.grade || "--";
  dom.grade.textContent = grade.toUpperCase();
  dom.grade.className = `grade-badge grade-${grade}`;
  dom.confidence.textContent = data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : "--";
  const dev = data.deviations || {};
  dom.vVel.textContent = dev.velocity_pct != null ? `${dev.velocity_pct.toFixed(2)}%` : "--";
  dom.vEarth.textContent = dev.earth_dist_pct != null ? `${dev.earth_dist_pct.toFixed(2)}%` : "--";
  dom.vMoon.textContent = dev.moon_dist_pct != null ? `${dev.moon_dist_pct.toFixed(2)}%` : "--";
}

function addAlert(alert) {
  alertBuffer.push(alert);
  if (alertBuffer.length > 20) alertBuffer = alertBuffer.slice(-15);
  const el = document.createElement("div");
  el.className = "alert-item";
  const type = (alert.alert_type || "UNKNOWN").toUpperCase().replace("_", " ");
  el.innerHTML = `<div class="alert-type type-${alert.alert_type || ""}">${type}</div>
    <div class="alert-detail">${alert.details || ""}</div>`;
  dom.alertList.prepend(el);
  while (dom.alertList.children.length > 8) dom.alertList.removeChild(dom.alertList.lastChild);
}

// ── WebSocket for live updates ──

let validationCount = 0;
function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);
  ws.onopen = () => { dom.connStatus.textContent = "LIVE"; dom.connStatus.className = "status-live"; };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "telemetry" && msg.data) {
        updateTelemetryDOM(msg.data);
        // If Horizons data with ECI, add to animation
        if (msg.data.pos_km) {
          const jd = JulianDate.fromDate(new Date(msg.data.timestamp * 1000));
          orionPosition.addSample(jd, eciToCartesian3(msg.data.pos_km, Cartesian3));
          // Extend clock stop time
          viewer.clock.stopTime = jd.clone();
        }
      }
      else if (msg.type === "alert") addAlert(msg.data);
      else if (msg.type === "validation") { updateValidation(msg.data); dom.vCount.textContent = ++validationCount; }
    } catch {}
  };
  ws.onclose = () => { dom.connStatus.textContent = "OFFLINE"; dom.connStatus.className = "status-disconnected"; setTimeout(connectWs, 3000); };
  ws.onerror = () => ws.close();
}
connectWs();
