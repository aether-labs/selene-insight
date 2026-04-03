/**
 * Open MCT Alert Plugin — Skeptic Agent alerts as notifications.
 */

export function AlertPlugin() {
  return function install(openmct) {
    let ws;

    function connect() {
      const protocol =
        window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(
        `${protocol}//${window.location.host}/ws/telemetry`
      );

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "alert" && msg.data) {
            showAlert(msg.data);
          }
        } catch {
          /* ignore */
        }
      };

      ws.onclose = () => {
        setTimeout(connect, 3000);
      };
    }

    function showAlert(alert) {
      const msg = `[${(alert.alert_type || "UNKNOWN").toUpperCase()}] ${alert.details || "Physics anomaly detected"}`;

      // Open MCT notification API varies by version
      if (openmct.notifications && openmct.notifications.alert) {
        openmct.notifications.alert(msg);
      } else if (openmct.notifications && openmct.notifications.notify) {
        openmct.notifications.notify({ message: msg, severity: "alert" });
      } else {
        console.warn("[SKEPTIC]", msg);
      }
    }

    openmct.on("start", () => {
      connect();
    });
  };
}
