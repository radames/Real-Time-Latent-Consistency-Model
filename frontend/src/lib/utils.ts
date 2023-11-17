export function LCMLive(webcamVideo, liveImage) {
    let websocket: WebSocket;

    async function start() {
        return new Promise((resolve, reject) => {
            const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                }:${window.location.host}/ws`;

            const socket = new WebSocket(websocketURL);
            socket.onopen = () => {
                console.log("Connected to websocket");
            };
            socket.onclose = () => {
                console.log("Disconnected from websocket");
                stop();
                resolve({ "status": "disconnected" });
            };
            socket.onerror = (err) => {
                console.error(err);
                reject(err);
            };
            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                switch (data.status) {
                    case "success":
                        break;
                    case "start":
                        const userId = data.userId;
                        initVideoStream(userId);
                        break;
                    case "timeout":
                        stop();
                        resolve({ "status": "timeout" });
                    case "error":
                        stop();
                        reject(data.message);

                }
            };
            websocket = socket;
        })
    }
    function switchCamera() {
        const constraints = {
            audio: false,
            video: { width: 1024, height: 1024, deviceId: mediaDevices[webcamsEl.value].deviceId }
        };
        navigator.mediaDevices
            .getUserMedia(constraints)
            .then((mediaStream) => {
                webcamVideo.removeEventListener("timeupdate", videoTimeUpdateHandler);
                webcamVideo.srcObject = mediaStream;
                webcamVideo.onloadedmetadata = () => {
                    webcamVideo.play();
                    webcamVideo.addEventListener("timeupdate", videoTimeUpdateHandler);
                };
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
            });
    }

    async function videoTimeUpdateHandler() {
        const dimension = getValue("input[name=dimension]:checked");
        const [WIDTH, HEIGHT] = JSON.parse(dimension);

        const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
        const videoW = webcamVideo.videoWidth;
        const videoH = webcamVideo.videoHeight;
        const aspectRatio = WIDTH / HEIGHT;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(webcamVideo, videoW / 2 - videoH * aspectRatio / 2, 0, videoH * aspectRatio, videoH, 0, 0, WIDTH, HEIGHT)
        const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 1 });
        websocket.send(blob);
        websocket.send(JSON.stringify({
            "seed": getValue("#seed"),
            "prompt": getValue("#prompt"),
            "guidance_scale": getValue("#guidance-scale"),
            "strength": getValue("#strength"),
            "steps": getValue("#steps"),
            "lcm_steps": getValue("#lcm_steps"),
            "width": WIDTH,
            "height": HEIGHT,
            "controlnet_scale": getValue("#controlnet_scale"),
            "controlnet_start": getValue("#controlnet_start"),
            "controlnet_end": getValue("#controlnet_end"),
            "canny_low_threshold": getValue("#canny_low_threshold"),
            "canny_high_threshold": getValue("#canny_high_threshold"),
            "debug_canny": getValue("#debug_canny")
        }));
    }
    let mediaDevices = [];
    async function initVideoStream(userId) {
        liveImage.src = `/stream/${userId}`;
        await navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const cameras = devices.filter(device => device.kind === 'videoinput');
                mediaDevices = cameras;
                webcamsEl.innerHTML = "";
                cameras.forEach((camera, index) => {
                    const option = document.createElement("option");
                    option.value = index;
                    option.innerText = camera.label;
                    webcamsEl.appendChild(option);
                    option.selected = index === 0;
                });
                webcamsEl.addEventListener("change", switchCamera);
            })
            .catch(err => {
                console.error(err);
            });
        const constraints = {
            audio: false,
            video: { width: 1024, height: 1024, deviceId: mediaDevices[0].deviceId }
        };
        navigator.mediaDevices
            .getUserMedia(constraints)
            .then((mediaStream) => {
                webcamVideo.srcObject = mediaStream;
                webcamVideo.onloadedmetadata = () => {
                    webcamVideo.play();
                    webcamVideo.addEventListener("timeupdate", videoTimeUpdateHandler);
                };
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
            });
    }


    async function stop() {
        websocket.close();
        navigator.mediaDevices.getUserMedia({ video: true }).then((mediaStream) => {
            mediaStream.getTracks().forEach((track) => track.stop());
        });
        webcamVideo.removeEventListener("timeupdate", videoTimeUpdateHandler);
        webcamsEl.removeEventListener("change", switchCamera);
        webcamVideo.srcObject = null;
    }
    return {
        start,
        stop
    }
}