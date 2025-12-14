"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface ProgressEvent {
  timestamp: string;
  event_type: string;
  data?: {
    system_name?: string;
    processed?: number;
    total?: number;
    phase_name?: string;
    status?: string;
    [key: string]: unknown;
  };
  level?: string;
}

interface UseSignalProgressReturn {
  isConnected: boolean;
  events: ProgressEvent[];
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  clearEvents: () => void;
}

export function useSignalProgress(): UseSignalProgressReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    // Prevent multiple connections
    if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
      return;
    }

    try {
      console.log("[WS] Connecting to WebSocket...");
      const ws = new WebSocket("ws://localhost:8000/ws/signals/progress");

      ws.onopen = () => {
        console.log("[WS] Connected!");
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as ProgressEvent;
          console.log("[WS] Received:", data.event_type);
          setEvents((prev) => [...prev.slice(-100), data]); // Keep last 100 events
        } catch {
          // Ignore parse errors
        }
      };

      ws.onerror = (e) => {
        console.error("[WS] Error:", e);
        setError("WebSocket connection error");
      };

      ws.onclose = (e) => {
        console.log("[WS] Closed:", e.code, e.reason);
        setIsConnected(false);
        wsRef.current = null;
      };

      wsRef.current = ws;
    } catch (err) {
      console.error("[WS] Failed to connect:", err);
      setError(`Failed to connect: ${err}`);
    }
  }, []);

  const disconnect = useCallback(() => {
    console.log("[WS] Disconnecting...");
    if (wsRef.current) {
      wsRef.current.close(1000, "Client disconnect");
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, "Component unmount");
      }
    };
  }, []);

  return {
    isConnected,
    events,
    error,
    connect,
    disconnect,
    clearEvents,
  };
}
