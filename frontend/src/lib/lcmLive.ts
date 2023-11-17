import { writable } from 'svelte/store';
import { PUBLIC_BASE_URL, PUBLIC_WSS_URL } from '$env/static/public';

export const isStreaming = writable(false);
export const isLCMRunning = writable(false);


export enum LCMLiveStatus {
    INIT = "init",
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
}

interface lcmLive {
    streamId: string | null;
    status: LCMLiveStatus
}

const initialState: lcmLive = {
    streamId: null,
    status: LCMLiveStatus.INIT
};

export const lcmLiveState = writable(initialState);

let websocket: WebSocket | null = null;
export const lcmLiveActions = {
    async start() {

        isLCMRunning.set(true);
        try {
            const websocketURL = PUBLIC_WSS_URL ? PUBLIC_WSS_URL : `${window.location.protocol === "https:" ? "wss" : "ws"
                }:${window.location.host}/ws`;

            websocket = new WebSocket(websocketURL);
            websocket.onopen = () => {
                console.log("Connected to websocket");
            };
            websocket.onclose = () => {
                lcmLiveState.update((state) => ({
                    ...state,
                    status: LCMLiveStatus.DISCONNECTED
                }));
                console.log("Disconnected from websocket");
                isLCMRunning.set(false);
            };
            websocket.onerror = (err) => {
                console.error(err);
            };
            websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("WS: ", data);
                switch (data.status) {
                    case "success":
                        break;
                    case "start":
                        const streamId = data.userId;
                        lcmLiveState.update((state) => ({
                            ...state,
                            status: LCMLiveStatus.CONNECTED,
                            streamId: streamId,
                        }));
                        break;
                    case "timeout":
                        console.log("timeout");
                    case "error":
                        console.log(data.message);
                        isLCMRunning.set(false);
                }
            };
            lcmLiveState.update((state) => ({
                ...state,
            }));
        } catch (err) {
            console.error(err);
            isLCMRunning.set(false);
        }
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {

        if (websocket) {
            websocket.close();
        }
        websocket = null;
        lcmLiveState.set({ status: LCMLiveStatus.DISCONNECTED, streamId: null });
        isLCMRunning.set(false)
    },
};