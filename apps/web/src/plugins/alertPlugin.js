/**
 * Open MCT Alert Plugin — Skeptic Agent alerts as notifications.
 *
 * Shows physics verification alerts in the Open MCT notification area
 * and provides a list view of recent alerts.
 */

export function AlertPlugin() {
  return function install(openmct) {
    // Connect WebSocket for alerts
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
      const typeMap = {
        orbital_maneuver: "caution",
        sensor_anomaly: "alert",
        coast_nominal: "info",
      };

      openmct.notifications.notify({
        message: `[${alert.alert_type?.toUpperCase()}] ${alert.details || "Physics anomaly detected"}`,
        severity: typeMap[alert.alert_type] || "info",
        autoDismiss: true,
        autoDismissTimeout: 15000,
      });
    }

    // Start on app ready
    openmct.on("start", () => {
      connect();

      // Load existing alerts
      fetch("/api/alerts/latest?n=10")
        .then((r) => r.json())
        .then((d) => {
          if (d.data) {
            d.data.slice(0, 3).forEach(showAlert);
          }
        })
        .catch(() => {});
    });
  };
}
