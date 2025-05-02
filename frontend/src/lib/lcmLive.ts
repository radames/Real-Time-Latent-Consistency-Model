import { get, writable } from "svelte/store";

export enum LCMLiveStatus {
  CONNECTED = "connected",
  DISCONNECTED = "disconnected",
  CONNECTING = "connecting",
  WAIT = "wait",
  SEND_FRAME = "send_frame",
  TIMEOUT = "timeout",
  ERROR = "error",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

// WebSocket connection
let websocket: WebSocket | null;

// Register browser unload event listener to properly close WebSockets
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    // Close the WebSocket properly if it exists
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.close(1000, "Page unload");
    }
  });
}
export const lcmLiveActions = {
  async start(
    getSreamdata: () =>
      | [Record<string, unknown>]
      | [Record<string, unknown>, Blob],
  ) {
    return new Promise((resolve, reject) => {
      try {
        // Set connecting status immediately
        lcmLiveStatus.set(LCMLiveStatus.CONNECTING);

        const userId = crypto.randomUUID();
        const websocketURL = `${
          window.location.protocol === "https:" ? "wss" : "ws"
        }:${window.location.host}/api/ws/${userId}`;

        // Close any existing connection first
        if (websocket && websocket.readyState !== WebSocket.CLOSED) {
          websocket.close();
        }

        websocket = new WebSocket(websocketURL);

        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
          if (websocket && websocket.readyState !== WebSocket.OPEN) {
            console.error("WebSocket connection timeout");
            lcmLiveStatus.set(LCMLiveStatus.ERROR);
            streamId.set(null);
            reject(new Error("Connection timeout. Please try again."));
            websocket.close();
          }
        }, 10000); // 10 second timeout

        websocket.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log("Connected to websocket");
        };

        websocket.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log(
            `Disconnected from websocket: ${event.code} ${event.reason}`,
          );

          // Only change status if we're not in ERROR state (which would mean we already handled the error)
          if (get(lcmLiveStatus) !== LCMLiveStatus.ERROR) {
            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
          }

          // If connection was never established (close without open)
          if (event.code === 1006 && get(streamId) === null) {
            reject(
              new Error("Cannot connect to server. Please try again later."),
            );
          }
        };

        websocket.onerror = (err) => {
          clearTimeout(connectionTimeout);
          console.error("WebSocket error:", err);
          lcmLiveStatus.set(LCMLiveStatus.ERROR);
          streamId.set(null);
          reject(new Error("Connection error. Please try again."));
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            switch (data.status) {
              case "connected":
                lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                streamId.set(userId);
                resolve({ status: "connected", userId });
                break;
              case "send_frame":
                lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                try {
                  const streamData = getSreamdata();
                  // Send as an object, not a string, to use the proper handling in the send method
                  this.send({ status: "next_frame" });
                  for (const d of streamData) {
                    this.send(d);
                  }
                } catch (error) {
                  console.error("Error sending frame data:", error);
                }
                break;
              case "wait":
                lcmLiveStatus.set(LCMLiveStatus.WAIT);
                break;
              case "timeout":
                console.log("Session timeout");
                lcmLiveStatus.set(LCMLiveStatus.TIMEOUT);
                streamId.set(null);
                reject(new Error("Session timeout. Please restart."));
                break;
              case "error":
                console.error("Server error:", data.message);
                lcmLiveStatus.set(LCMLiveStatus.ERROR);
                streamId.set(null);
                reject(new Error(data.message || "Server error occurred"));
                break;
              default:
                console.log("Unknown message status:", data.status);
            }
          } catch (error) {
            console.error("Error handling websocket message:", error);
          }
        };
      } catch (err) {
        console.error("Error initializing websocket:", err);
        lcmLiveStatus.set(LCMLiveStatus.ERROR);
        streamId.set(null);
        reject(err);
      }
    });
  },
  send(data: Blob | Record<string, unknown>) {
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        if (data instanceof Blob) {
          websocket.send(data);
        } else {
          websocket.send(JSON.stringify(data));
        }
      } else {
        const readyStateText = websocket
          ? ["CONNECTING", "OPEN", "CLOSING", "CLOSED"][websocket.readyState]
          : "null";
        console.warn(`WebSocket not ready for sending: ${readyStateText}`);

        // If WebSocket is closed unexpectedly, set status to disconnected
        if (!websocket || websocket.readyState === WebSocket.CLOSED) {
          lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
          streamId.set(null);
        }
      }
    } catch (error) {
      console.error("Error sending data through WebSocket:", error);
      // Handle WebSocket error by forcing disconnection
      this.stop();
    }
  },

  async reconnect(
    getSreamdata: () =>
      | [Record<string, unknown>]
      | [Record<string, unknown>, Blob],
  ) {
    try {
      await this.stop();
      // Small delay to ensure clean disconnection before reconnecting
      await new Promise((resolve) => setTimeout(resolve, 500));
      return await this.start(getSreamdata);
    } catch (error) {
      console.error("Reconnection failed:", error);
      throw error;
    }
  },

  async stop() {
    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
    try {
      if (websocket) {
        // Only attempt to close if not already closed
        if (websocket.readyState !== WebSocket.CLOSED) {
          // Set up onclose handler to clean up only
          websocket.onclose = () => {
            console.log("WebSocket closed cleanly during stop()");
          };

          // Set up onerror to be silent during intentional closure
          websocket.onerror = () => {};

          websocket.close(1000, "Client initiated disconnect");
        }
      }
    } catch (error) {
      console.error("Error during WebSocket closure:", error);
    } finally {
      // Always clean up references
      websocket = null;
      streamId.set(null);
    }
  },
};
