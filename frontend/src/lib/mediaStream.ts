import { writable, type Writable } from 'svelte/store';

export enum MediaStreamStatus {
    INIT = "init",
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
}
export const onFrameChangeStore: Writable<{ now: Number, metadata: VideoFrameCallbackMetadata, blob: Blob }> = writable();
export const isMediaStreaming = writable(MediaStreamStatus.INIT);

interface mediaStream {
    mediaStream: MediaStream | null;
    status: MediaStreamStatus
    devices: MediaDeviceInfo[];
}

const initialState: mediaStream = {
    mediaStream: null,
    status: MediaStreamStatus.INIT,
    devices: [],
};

export const mediaStreamState = writable(initialState);

export const mediaStreamActions = {
    async enumerateDevices() {
        console.log("Enumerating devices");
        await navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const cameras = devices.filter(device => device.kind === 'videoinput');
                console.log("Cameras: ", cameras);
                mediaStreamState.update((state) => ({
                    ...state,
                    devices: cameras,
                }));
            })
            .catch(err => {
                console.error(err);
            });
    },
    async start(mediaDevicedID?: string) {
        const constraints = {
            audio: false,
            video: {
                width: 1024, height: 1024, deviceId: mediaDevicedID
            }
        };

        await navigator.mediaDevices
            .getUserMedia(constraints)
            .then((mediaStream) => {
                mediaStreamState.update((state) => ({
                    ...state,
                    mediaStream: mediaStream,
                    status: MediaStreamStatus.CONNECTED,
                }));
                isMediaStreaming.set(MediaStreamStatus.CONNECTED);
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
                isMediaStreaming.set(MediaStreamStatus.DISCONNECTED);
            });
    },
    async switchCamera(mediaDevicedID: string) {
        const constraints = {
            audio: false,
            video: { width: 1024, height: 1024, deviceId: mediaDevicedID }
        };
        await navigator.mediaDevices
            .getUserMedia(constraints)
            .then((mediaStream) => {
                mediaStreamState.update((state) => ({
                    ...state,
                    mediaStream: mediaStream,
                    status: MediaStreamStatus.CONNECTED,
                }));
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
            });
    },
    async stop() {
        navigator.mediaDevices.getUserMedia({ video: true }).then((mediaStream) => {
            mediaStream.getTracks().forEach((track) => track.stop());
        });
        mediaStreamState.update((state) => ({
            ...state,
            mediaStream: null,
            status: MediaStreamStatus.DISCONNECTED,
        }));
        isMediaStreaming.set(MediaStreamStatus.DISCONNECTED);
    },
};