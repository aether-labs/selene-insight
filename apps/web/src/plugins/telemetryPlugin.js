/**
 * Open MCT Telemetry Plugin — connects to the Selene-Insight API.
 *
 * Provides:
 * - Domain object tree (Artemis II → telemetry points)
 * - Historical telemetry provider (REST)
 * - Real-time telemetry provider (WebSocket)
 */

const NAMESPACE = "selene";
const ROOT_KEY = "artemis-ii";

const TELEMETRY_KEYS = [
  {
    key: "velocity_kms",
    name: "Velocity",
    units: "km/s",
    format: "float",
    hints: { range: 1 },
  },
  {
    key: "earth_dist_km",
    name: "Earth Distance",
    units: "km",
    format: "float",
    hints: { range: 2 },
  },
  {
    key: "moon_dist_km",
    name: "Moon Distance",
    units: "km",
    format: "float",
    hints: { range: 3 },
  },
];

export function TelemetryPlugin() {
  return function install(openmct) {
    // --- Root object ---
    const rootObj = {
      identifier: { namespace: NAMESPACE, key: ROOT_KEY },
      name: "Artemis II",
      type: "folder",
      location: "ROOT",
    };

    openmct.objects.addRoot({
      namespace: NAMESPACE,
      key: ROOT_KEY,
    });

    // --- Object provider ---
    openmct.objects.addProvider(NAMESPACE, {
      get(identifier) {
        if (identifier.key === ROOT_KEY) {
          return Promise.resolve({
            ...rootObj,
            composition: TELEMETRY_KEYS.map((t) => ({
              namespace: NAMESPACE,
              key: t.key,
            })),
          });
        }

        const meta = TELEMETRY_KEYS.find((t) => t.key === identifier.key);
        if (meta) {
          return Promise.resolve({
            identifier,
            name: meta.name,
            type: "selene.telemetry",
            telemetry: {
              values: [
                {
                  key: "utc",
                  source: "timestamp",
                  name: "Timestamp",
                  format: "utc",
                  hints: { domain: 1 },
                },
                {
                  key: "value",
                  source: meta.key,
                  name: meta.name,
                  units: meta.units,
                  format: meta.format,
                  hints: meta.hints,
                },
              ],
            },
            location: `${NAMESPACE}:${ROOT_KEY}`,
          });
        }

        return Promise.reject();
      },
    });

    // --- Composition provider ---
    openmct.composition.addProvider({
      appliesTo(domainObject) {
        return (
          domainObject.identifier.namespace === NAMESPACE &&
          domainObject.identifier.key === ROOT_KEY
        );
      },
      load() {
        return Promise.resolve(
          TELEMETRY_KEYS.map((t) => ({
            namespace: NAMESPACE,
            key: t.key,
          }))
        );
      },
    });

    // --- Type ---
    openmct.types.addType("selene.telemetry", {
      name: "Selene Telemetry",
      description: "Artemis II telemetry data point",
      cssClass: "icon-telemetry",
    });

    // --- Historical telemetry provider ---
    openmct.telemetry.addProvider({
      supportsRequest(domainObject) {
        return domainObject.type === "selene.telemetry";
      },
      request(domainObject, options) {
        const start = (options.start || Date.now() - 3600000) / 1000;
        const end = (options.end || Date.now()) / 1000;

        return fetch(`/api/telemetry/range?start=${start}&end=${end}&limit=5000`)
          .then((r) => r.json())
          .then((d) =>
            (d.data || []).map((pt) => ({
              ...pt,
              timestamp: pt.timestamp * 1000, // Open MCT expects ms
            }))
          );
      },
    });

    // --- Real-time telemetry provider (WebSocket) ---
    openmct.telemetry.addProvider({
      supportsSubscribe(domainObject) {
        return domainObject.type === "selene.telemetry";
      },
      subscribe(domainObject, callback) {
        const key = domainObject.identifier.key;
        const protocol =
          window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(
          `${protocol}//${window.location.host}/ws/telemetry`
        );

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "telemetry" && msg.data) {
              callback({
                ...msg.data,
                timestamp: msg.data.timestamp * 1000,
              });
            }
          } catch {
            /* ignore */
          }
        };

        return function unsubscribe() {
          ws.close();
        };
      },
    });
  };
}
