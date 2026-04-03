/**
 * Open MCT CesiumJS View Plugin — 3D Earth-Moon-Orion visualization.
 *
 * Renders as an Open MCT view that can be placed in any layout.
 */

import {
  Viewer as CesiumViewer,
  Cartesian3,
  Color,
  Ion,
  LabelStyle,
  VerticalOrigin,
  NearFarScalar,
  PolylineDashMaterialProperty,
  CallbackProperty,
  Cartesian2,
  SceneMode,
} from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

import { telemetryToPosition, moonToPosition } from "../lib/orbit.js";
import {
  generateReferenceTrajectory,
  estimateLaunchTime,
} from "../lib/referenceTrajectory.js";

Ion.defaultAccessToken =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc1ODcsImlhdCI6MTYyNzg0NTE4Mn0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk";

const MOON_RADIUS_M = 1.737e6;

export function CesiumViewPlugin() {
  return function install(openmct) {
    openmct.types.addType("selene.cesium-view", {
      name: "3D Trajectory View",
      description: "Earth-Moon-Orion 3D visualization",
      cssClass: "icon-object",
      creatable: true,
    });

    openmct.objectViews.addProvider({
      key: "selene.cesium",
      name: "3D Trajectory",
      cssClass: "icon-object",

      canView(domainObject) {
        return domainObject.type === "selene.cesium-view";
      },

      view(domainObject) {
        let container;
        let viewer;
        let ws;
        let telemetryBuffer = [];
        let refTrajectoryEntity;
        let actualPathEntity;
        let orionEntity;
        let orionGlowEntity;
        let moonEntity;
        let earthLabelEntity;
        let launchTime = null;

        return {
          show(el) {
            container = document.createElement("div");
            container.style.cssText =
              "width:100%;height:100%;position:relative;";
            el.appendChild(container);

            viewer = new CesiumViewer(container, {
              timeline: false,
              animation: false,
              homeButton: false,
              geocoder: false,
              sceneModePicker: false,
              baseLayerPicker: false,
              navigationHelpButton: false,
              fullscreenButton: false,
              infoBox: true,
              selectionIndicator: true,
              sceneMode: SceneMode.SCENE3D,
            });

            // Remove default imagery credit
            viewer.cesiumWidget.creditContainer.style.display = "none";

            // Initial camera: show Earth-Moon system
            viewer.camera.setView({
              destination: Cartesian3.fromDegrees(0, 10, 600_000_000),
            });

            // --- Static entities ---

            // Reference trajectory (will be updated when we get MET)
            refTrajectoryEntity = viewer.entities.add({
              name: "Predicted Trajectory",
              polyline: {
                positions: [],
                width: 2,
                material: new PolylineDashMaterialProperty({
                  color: Color.fromCssColorString("#ffffff").withAlpha(0.25),
                  dashLength: 16,
                }),
              },
            });

            // Actual telemetry path
            actualPathEntity = viewer.entities.add({
              name: "Actual Path",
              polyline: {
                positions: new CallbackProperty(() => {
                  return telemetryBuffer
                    .map((p) => telemetryToPosition(p, Cartesian3))
                    .filter(Boolean);
                }, false),
                width: 4,
                material: Color.fromCssColorString("#00ccff").withAlpha(0.9),
              },
            });

            // Orion glow
            orionGlowEntity = viewer.entities.add({
              name: "Orion Glow",
              position: Cartesian3.ZERO,
              point: {
                pixelSize: 30,
                color: Color.fromCssColorString("#00ccff").withAlpha(0.25),
              },
            });

            // Orion
            orionEntity = viewer.entities.add({
              name: "Orion",
              position: Cartesian3.ZERO,
              point: {
                pixelSize: 12,
                color: Color.CYAN,
                outlineColor: Color.WHITE,
                outlineWidth: 1,
              },
              label: {
                text: "ORION",
                font: "bold 16px monospace",
                fillColor: Color.CYAN,
                style: LabelStyle.FILL_AND_OUTLINE,
                outlineColor: Color.BLACK,
                outlineWidth: 3,
                verticalOrigin: VerticalOrigin.BOTTOM,
                pixelOffset: new Cartesian2(0, -22),
                scaleByDistance: new NearFarScalar(5e5, 1.2, 5e8, 0.5),
              },
            });

            // Moon
            const nowTs = Date.now() / 1000;
            const moonPos = moonToPosition(nowTs, Cartesian3);
            moonEntity = viewer.entities.add({
              name: "Moon",
              position: moonPos,
              ellipsoid: {
                radii: new Cartesian3(
                  MOON_RADIUS_M,
                  MOON_RADIUS_M,
                  MOON_RADIUS_M
                ),
                material: Color.fromCssColorString("#dddddd").withAlpha(0.95),
              },
              label: {
                text: "MOON",
                font: "bold 16px monospace",
                fillColor: Color.fromCssColorString("#dddddd"),
                style: LabelStyle.FILL_AND_OUTLINE,
                outlineColor: Color.BLACK,
                outlineWidth: 3,
                verticalOrigin: VerticalOrigin.BOTTOM,
                pixelOffset: new Cartesian2(0, -24),
                scaleByDistance: new NearFarScalar(5e5, 1.2, 5e8, 0.5),
              },
            });

            // Earth label
            earthLabelEntity = viewer.entities.add({
              name: "Earth",
              position: Cartesian3.fromDegrees(0, 90, 0),
              label: {
                text: "EARTH",
                font: "bold 16px monospace",
                fillColor: Color.fromCssColorString("#4488ff"),
                style: LabelStyle.FILL_AND_OUTLINE,
                outlineColor: Color.BLACK,
                outlineWidth: 3,
                verticalOrigin: VerticalOrigin.BOTTOM,
                pixelOffset: new Cartesian2(0, -12),
                scaleByDistance: new NearFarScalar(5e5, 1.2, 5e8, 0.5),
              },
            });

            // --- Fetch initial data ---
            fetch("/api/telemetry/latest?n=500")
              .then((r) => r.json())
              .then((d) => {
                if (d.data && d.data.length) {
                  telemetryBuffer = d.data;
                  updateFromBuffer();
                }
              })
              .catch(() => {});

            // --- WebSocket for live updates ---
            connectWs();
          },

          destroy() {
            if (ws) ws.close();
            if (viewer) viewer.destroy();
            if (container && container.parentNode) {
              container.parentNode.removeChild(container);
            }
          },
        };

        function connectWs() {
          const protocol =
            window.location.protocol === "https:" ? "wss:" : "ws:";
          ws = new WebSocket(
            `${protocol}//${window.location.host}/ws/telemetry`
          );

          ws.onmessage = (evt) => {
            try {
              const msg = JSON.parse(evt.data);
              if (msg.type === "telemetry" && msg.data) {
                telemetryBuffer.push(msg.data);
                if (telemetryBuffer.length > 2000) {
                  telemetryBuffer = telemetryBuffer.slice(-1500);
                }
                updateFromBuffer();
              }
            } catch {
              /* ignore */
            }
          };

          ws.onclose = () => {
            setTimeout(connectWs, 3000);
          };
        }

        function updateFromBuffer() {
          const latest = telemetryBuffer[telemetryBuffer.length - 1];
          if (!latest) return;

          // Update Orion position
          const pos = telemetryToPosition(latest, Cartesian3);
          if (pos) {
            orionEntity.position = pos;
            orionGlowEntity.position = pos;
          }

          // Update Moon position
          const ts = latest.timestamp || Date.now() / 1000;
          const moonPos = moonToPosition(ts, Cartesian3);
          moonEntity.position = moonPos;

          // Generate reference trajectory on first data
          if (!launchTime) {
            launchTime = estimateLaunchTime(latest);
            if (launchTime) {
              const refPositions = generateReferenceTrajectory(
                launchTime,
                Cartesian3
              );
              refTrajectoryEntity.polyline.positions = refPositions;
            }
          }
        }
      },
    });

    // --- Create a default instance in the root ---
    const cesiumObj = {
      identifier: { namespace: "selene", key: "cesium-3d" },
      name: "3D Trajectory View",
      type: "selene.cesium-view",
      location: "selene:artemis-ii",
    };

    openmct.objects.addProvider("selene", {
      get(identifier) {
        if (identifier.key === "cesium-3d") {
          return Promise.resolve(cesiumObj);
        }
        return undefined;
      },
    });
  };
}
