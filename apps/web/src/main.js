/**
 * Selene-Insight — Full-screen CesiumJS dashboard.
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
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { telemetryToPosition, moonToPosition, eciToCartesian3 } from "./lib/orbit.js";
import {
  generateReferenceTrajectory,
  generateMoonOrbit,
  estimateLaunchTime,
} from "./lib/referenceTrajectory.js";

// ── Config ──
Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

const MOON_RADIUS_M = 1.737e6;
const LABEL_FONT = "bold 22px monospace";

// ── State ──
let telemetryBuffer = [];
let alertBuffer = [];
let launchTime = null;

// ── CesiumJS viewer ──
const viewer = new Viewer("cesium-container", {
  timeline: false,
  animation: false,
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

// ── Entities ──

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

// Predicted trajectory (dashed white) — from distance model
const refTrajectoryEntity = viewer.entities.add({
  name: "Predicted Trajectory",
  polyline: {
    positions: [],
    width: 2,
    material: new PolylineDashMaterialProperty({
      color: Color.fromCssColorString("#ffffff").withAlpha(0.35),
      dashLength: 20,
    }),
  },
});

// Actual flown path from Horizons (solid cyan) — full history
const historyPathEntity = viewer.entities.add({
  name: "Flown Path (Horizons)",
  polyline: {
    positions: [],
    width: 4,
    material: Color.fromCssColorString("#00ccff").withAlpha(0.9),
  },
});

// Live scraper path (thinner, builds in real-time)
const livePathEntity = viewer.entities.add({
  name: "Live Path",
  polyline: {
    positions: [],
    width: 2,
    material: Color.fromCssColorString("#00ffaa").withAlpha(0.6),
  },
});

// Orion glow
const orionGlow = viewer.entities.add({
  position: Cartesian3.ZERO,
  point: { pixelSize: 36, color: Color.fromCssColorString("#00ccff").withAlpha(0.2) },
});

// Orion
const orionEntity = viewer.entities.add({
  name: "Orion",
  position: Cartesian3.ZERO,
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

// Earth — visible sphere (3x exaggerated)
const EARTH_VIS_RADIUS = 6371000 * 3;
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
    pixelOffset: new Cartesian2(0, -20),
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

// ── Update functions ──

let livePositions = [];

function updateTelemetry(point) {
  telemetryBuffer.push(point);
  if (telemetryBuffer.length > 2000) telemetryBuffer = telemetryBuffer.slice(-1500);

  // Update Orion position
  const pos = telemetryToPosition(point, Cartesian3);
  if (pos) {
    orionEntity.position = pos;
    orionGlow.position = pos;
    livePositions.push(pos);
    if (livePositions.length > 1) {
      livePathEntity.polyline.positions = [...livePositions];
    }
  }

  // Update Moon
  const ts = point.timestamp || Date.now() / 1000;
  moonEntity.position = moonToPosition(ts, Cartesian3);

  // Generate predicted trajectory once
  if (!launchTime) {
    launchTime = estimateLaunchTime(point);
    if (launchTime) {
      refTrajectoryEntity.polyline.positions = generateReferenceTrajectory(launchTime, Cartesian3);
    }
  }

  // Update DOM
  dom.met.textContent = point.met || "--";
  dom.phase.textContent = point.phase || "--";
  dom.velocity.textContent = point.velocity_kms ? `${point.velocity_kms.toFixed(3)} km/s` : "--";
  dom.earth.textContent = point.earth_dist_km ? `${Math.round(point.earth_dist_km).toLocaleString()} km` : "--";
  dom.moon.textContent = point.moon_dist_km ? `${Math.round(point.moon_dist_km).toLocaleString()} km` : "--";
  dom.source.textContent = point.source || "issinfo";
  dom.count.textContent = telemetryBuffer.length;
}

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

// ── Fetch initial data ──

// 1. Full mission history from Horizons (solid cyan line: actual flown path)
fetch("/api/telemetry/history")
  .then((r) => r.json())
  .then((d) => {
    if (d.data && d.data.length) {
      const positions = d.data
        .filter((p) => p.pos_km)
        .map((p) => eciToCartesian3(p.pos_km, Cartesian3));
      if (positions.length > 1) {
        historyPathEntity.polyline.positions = positions;
      }
      console.log(`[HISTORY] Drew ${positions.length} points from Horizons`);
    }
  })
  .catch((e) => console.warn("[HISTORY] Failed:", e));

// 2. Recent live telemetry
fetch("/api/telemetry/latest?n=500")
  .then((r) => r.json())
  .then((d) => { if (d.data) d.data.reverse().forEach(updateTelemetry); })
  .catch(() => {});

// 3. Alerts
fetch("/api/alerts/latest?n=10")
  .then((r) => r.json())
  .then((d) => { if (d.data) d.data.reverse().forEach(addAlert); })
  .catch(() => {});

// 4. Validation
fetch("/api/validation/latest")
  .then((r) => r.json())
  .then((d) => {
    if (d.recent && d.recent.length) updateValidation(d.recent[d.recent.length - 1]);
    if (d.stats) dom.vCount.textContent = d.stats.total_validations || 0;
  })
  .catch(() => {});

// ── WebSocket ──
let validationCount = 0;
function connectWs() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/telemetry`);
  ws.onopen = () => { dom.connStatus.textContent = "LIVE"; dom.connStatus.className = "status-live"; };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "telemetry") updateTelemetry(msg.data);
      else if (msg.type === "alert") addAlert(msg.data);
      else if (msg.type === "validation") { updateValidation(msg.data); dom.vCount.textContent = ++validationCount; }
    } catch {}
  };
  ws.onclose = () => { dom.connStatus.textContent = "OFFLINE"; dom.connStatus.className = "status-disconnected"; setTimeout(connectWs, 3000); };
  ws.onerror = () => ws.close();
}
connectWs();
