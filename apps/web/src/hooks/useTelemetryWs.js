import { useEffect, useRef } from "react";

/**
 * WebSocket hook for live telemetry and alert streaming.
 */
export function useTelemetryWs({ onTelemetry, onAlert, enabled = true }) {
  const wsRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/telemetry`;

    let reconnectTimer;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[WS] Connected");
      };

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "telemetry" && onTelemetry) {
            onTelemetry(msg.data);
          } else if (msg.type === "alert" && onAlert) {
            onAlert(msg.data);
          }
        } catch {
          /* ignore malformed */
        }
      };

      ws.onclose = () => {
        console.log("[WS] Disconnected, reconnecting in 3s...");
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, onTelemetry, onAlert]);
}
