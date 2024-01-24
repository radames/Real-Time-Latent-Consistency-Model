import { writable, type Writable, type Readable, get, derived } from 'svelte/store';

const BASE_HEIGHT = 720;
export enum MediaStreamStatusEnum {
    INIT = "init",
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
}
export const onFrameChangeStore: Writable<{ blob: Blob }> = writable({ blob: new Blob() });

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
    async start(mediaDevicedID?: string, aspectRatio: number = 1) {
        const constraints = {
            audio: false,
            video: {
                width: {
                    ideal: BASE_HEIGHT * aspectRatio,
                },
                height: {
                    ideal: BASE_HEIGHT,
                },
                deviceId: mediaDevicedID
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
    async startScreenCapture() {
        const displayMediaOptions = {
            video: {
                displaySurface: "window",
            },
            audio: false,
            surfaceSwitching: "include"
        };


        let captureStream = null;

        try {
            captureStream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions);
            const videoTrack = captureStream.getVideoTracks()[0];

            console.log("Track settings:");
            console.log(JSON.stringify(videoTrack.getSettings(), null, 2));
            console.log("Track constraints:");
            console.log(JSON.stringify(videoTrack.getConstraints(), null, 2));
            mediaStreamStatus.set(MediaStreamStatusEnum.CONNECTED);
            mediaStream.set(captureStream)

            const capabilities = videoTrack.getCapabilities();
            const aspectRatio = capabilities.aspectRatio;
            console.log('Aspect Ratio Constraints:', aspectRatio);
        } catch (err) {
            console.error(err);
        }

    },
    async switchCamera(mediaDevicedID: string, aspectRatio: number) {
        console.log("Switching camera");
        if (get(mediaStreamStatus) !== MediaStreamStatusEnum.CONNECTED) {
            return;
        }
        const constraints = {
            audio: false,
            video: {
                width: {
                    ideal: BASE_HEIGHT * aspectRatio,
                },
                height: {
                    ideal: BASE_HEIGHT,
                },
                deviceId: mediaDevicedID
            }
        };
        console.log("Switching camera", constraints);
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