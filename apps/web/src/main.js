/**
 * Selene-Insight — Full-screen CesiumJS dashboard with time animation.
 *
 * Orion and Moon both animate along JPL Horizons trajectories.
 * Timeline covers full mission (launch → lunar flyby → return).
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
  SceneMode,
  JulianDate,
  ClockRange,
  ClockStep,
  SampledPositionProperty,
  LagrangePolynomialApproximation,
  PathGraphics,
  PolylineDashMaterialProperty,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { eciToCartesian3 } from "./lib/orbit.js";
import { generateMoonOrbit } from "./lib/referenceTrajectory.js";

// ── Config ──
Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

const MOON_RADIUS_M = 1.737e6;
const LABEL_FONT = "bold 22px monospace";

// ── State ──
let alertBuffer = [];

// ── Viewer ──
const viewer = new Viewer("cesium-container", {
  timeline: true,
  animation: true,
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

viewer.camera.setView({
  destination: Cartesian3.fromDegrees(20, 30, 800_000_000),
});

// ── Moon orbit ring (background) ──
const nowTs = Date.now() / 1000;
viewer.entities.add({
  name: "Moon Orbit",
  polyline: {
    positions: generateMoonOrbit(nowTs, Cartesian3),
    width: 1,
    material: Color.fromCssColorString("#444444").withAlpha(0.3),
  },
});

// ── Earth (1.5x exaggerated) ──
const EARTH_VIS_R = 6371000 * 1.5;
viewer.entities.add({
  name: "Earth",
  position: Cartesian3.fromDegrees(0, 0, 0),
  ellipsoid: {
    radii: new Cartesian3(EARTH_VIS_R, EARTH_VIS_R, EARTH_VIS_R),
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

// ── Orion (animated) ──
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
  path: new PathGraphics({
    leadTime: 0,
    trailTime: 86400 * 12,
    width: 3,
    material: Color.fromCssColorString("#00ccff").withAlpha(0.8),
  }),
});

// ── Orion predicted path (future, dashed) ──
const orionFuturePath = viewer.entities.add({
  name: "Orion Predicted",
  position: orionPosition,
  path: new PathGraphics({
    leadTime: 86400 * 12,
    trailTime: 0,
    width: 2,
    material: new PolylineDashMaterialProperty({
      color: Color.fromCssColorString("#00ccff").withAlpha(0.3),
      dashLength: 16,
    }),
  }),
});

// ── Moon (animated) ──
const moonPosition = new SampledPositionProperty();
moonPosition.setInterpolationOptions({
  interpolationDegree: 3,
  interpolationAlgorithm: LagrangePolynomialApproximation,
});

const moonEntity = viewer.entities.add({
  name: "Moon",
  position: moonPosition,
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

// ── Load full mission from Horizons ──

fetch("/api/telemetry/history")
  .then((r) => r.json())
  .then((d) => {
    if (!d.orion || !d.orion.length) return;

    // Load Orion trajectory
    for (const p of d.orion) {
      const jd = JulianDate.fromDate(new Date(p.timestamp * 1000));
      orionPosition.addSample(jd, eciToCartesian3(p.pos_km, Cartesian3));
    }

    // Load Moon trajectory
    if (d.moon) {
      for (const p of d.moon) {
        const jd = JulianDate.fromDate(new Date(p.timestamp * 1000));
        moonPosition.addSample(jd, eciToCartesian3(p.pos_km, Cartesian3));
      }
    }

    // Set clock to full mission
    const startJd = JulianDate.fromDate(new Date(d.orion[0].timestamp * 1000));
    const stopJd = JulianDate.fromDate(new Date(d.orion[d.orion.length - 1].timestamp * 1000));
    const nowJd = JulianDate.fromDate(new Date());

    viewer.clock.startTime = startJd.clone();
    viewer.clock.stopTime = stopJd.clone();
    viewer.clock.currentTime = nowJd.clone();
    viewer.clock.clockRange = ClockRange.LOOP_STOP;
    viewer.clock.clockStep = ClockStep.SYSTEM_CLOCK_MULTIPLIER;
    viewer.clock.multiplier = 1;

    viewer.timeline.zoomTo(startJd, stopJd);

    console.log(`[MISSION] Loaded ${d.orion.length} Orion + ${d.moon?.length || 0} Moon waypoints`);

    // Update telemetry panel periodically based on clock
    viewer.clock.onTick.addEventListener((clock) => {
      updateTelemetryFromClock(clock.currentTime, d.orion);
    });
  })
  .catch((e) => console.warn("[MISSION] Failed:", e));

// Update telemetry panel based on current animation time
let lastUpdateSec = 0;
function updateTelemetryFromClock(currentTime, orionData) {
  const nowSec = JulianDate.toDate(currentTime).getTime() / 1000;
  // Throttle to ~2 updates/sec
  if (Math.abs(nowSec - lastUpdateSec) < 0.5) return;
  lastUpdateSec = nowSec;

  // Find closest point
  let best = null;
  let bestDt = Infinity;
  for (const p of orionData) {
    const dt = Math.abs(p.timestamp - nowSec);
    if (dt < bestDt) { bestDt = dt; best = p; }
  }
  if (!best) return;

  // Compute distances from pos_km
  const [x, y, z] = best.pos_km;
  const earthDist = Math.sqrt(x * x + y * y + z * z);
  const vel = best.vel_kms
    ? Math.sqrt(best.vel_kms[0] ** 2 + best.vel_kms[1] ** 2 + best.vel_kms[2] ** 2)
    : 0;

  // Elapsed time from mission start
  const elapsed = nowSec - orionData[0].timestamp;
  const days = Math.floor(elapsed / 86400);
  const hrs = Math.floor((elapsed % 86400) / 3600);
  const mins = Math.floor((elapsed % 3600) / 60);
  const secs = Math.floor(elapsed % 60);

  const isFuture = nowSec > Date.now() / 1000;

  dom.met.textContent = `T+${days}d ${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  dom.phase.textContent = isFuture ? "Predicted" : earthDist > 300000 ? "Lunar Vicinity" : earthDist > 50000 ? "Outbound Coast" : "Earth Orbit";
  dom.velocity.textContent = vel ? `${vel.toFixed(3)} km/s` : "--";
  dom.earth.textContent = `${Math.round(earthDist).toLocaleString()} km`;
  dom.source.textContent = isFuture ? "prediction" : "jpl_horizons";
  dom.count.textContent = orionData.length;
}

// ── Alerts & Validation ──

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

// ── WebSocket ──
let validationCount = 0;
function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);
  ws.onopen = () => { dom.connStatus.textContent = "LIVE"; dom.connStatus.className = "status-live"; };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "alert") addAlert(msg.data);
      else if (msg.type === "validation") { updateValidation(msg.data); dom.vCount.textContent = ++validationCount; }
    } catch {}
  };
  ws.onclose = () => { dom.connStatus.textContent = "OFFLINE"; dom.connStatus.className = "status-disconnected"; setTimeout(connectWs, 3000); };
  ws.onerror = () => ws.close();
}
connectWs();
