import { writable, type Writable } from 'svelte/store';

export enum MediaStreamStatusEnum {
    INIT = "init",
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
}
export const onFrameChangeStore: Writable<{ now: Number, metadata: VideoFrameCallbackMetadata, blob: Blob }> = writable();

export const mediaDevices = writable<MediaDeviceInfo[]>([]);
export const mediaStreamStatus = writable(MediaStreamStatusEnum.INIT);
export const mediaStream = writable<MediaStream | null>(null);

export const mediaStreamActions = {
    async enumerateDevices() {
        // console.log("Enumerating devices");
        await navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const cameras = devices.filter(device => device.kind === 'videoinput');
                mediaDevices.set(cameras);
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
            .then((stream) => {
                mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
                mediaStream.set(stream);
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
                mediaStreamStatus.set(MediaStreamStatusEnum.DISCONNECTED);
                mediaStream.set(null);
            });
    },
    async switchCamera(mediaDevicedID: string) {
        const constraints = {
            audio: false,
            video: { width: 1024, height: 1024, deviceId: mediaDevicedID }
        };
        await navigator.mediaDevices
            .getUserMedia(constraints)
            .then((stream) => {
                mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
                mediaStream.set(stream)
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
            });
    },
    async stop() {
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            stream.getTracks().forEach((track) => track.stop());
        });
        mediaStreamStatus.set(MediaStreamStatusEnum.DISCONNECTED);
        mediaStream.set(null);
    },
};