from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import logging
import traceback
from config import Args
from user_queue import UserDataEventMap, UserDataEvent
import uuid
from asyncio import Event, sleep
import time
from PIL import Image
import io
from types import SimpleNamespace


def init_app(app: FastAPI, user_data_events: UserDataEventMap, args: Args, pipeline):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("Init app", app)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        if args.max_queue_size > 0 and len(user_data_events) >= args.max_queue_size:
            print("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            return

        try:
            uid = str(uuid.uuid4())
            print(f"New user connected: {uid}")
            await websocket.send_json(
                {"status": "success", "message": "Connected", "userId": uid}
            )
            user_data_events[uid] = UserDataEvent()
            print(f"User data events: {user_data_events}")
            await websocket.send_json(
                {"status": "start", "message": "Start Streaming", "userId": uid}
            )
            await handle_websocket_data(websocket, uid)
        except WebSocketDisconnect as e:
            logging.error(f"WebSocket Error: {e}, {uid}")
            traceback.print_exc()
        finally:
            print(f"User disconnected: {uid}")
            del user_data_events[uid]

    @app.get("/queue_size")
    async def get_queue_size():
        queue_size = len(user_data_events)
        return JSONResponse({"queue_size": queue_size})

    @app.get("/stream/{user_id}")
    async def stream(user_id: uuid.UUID):
        uid = str(user_id)
        try:

            async def generate():
                last_prompt: str = None
                while True:
                    data = await user_data_events[uid].wait_for_data()
                    params = data["params"]
                    # input_image = data["image"]
                    # if input_image is None:
                    # continue
                    image = pipeline.predict(params)
                    if image is None:
                        continue
                    frame_data = io.BytesIO()
                    image.save(frame_data, format="JPEG")
                    frame_data = frame_data.getvalue()
                    if frame_data is not None and len(frame_data) > 0:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"

                    await sleep(1.0 / 120.0)

            return StreamingResponse(
                generate(), media_type="multipart/x-mixed-replace;boundary=frame"
            )
        except Exception as e:
            logging.error(f"Streaming Error: {e}, {user_data_events}")
            traceback.print_exc()
            return HTTPException(status_code=404, detail="User not found")

    async def handle_websocket_data(websocket: WebSocket, user_id: uuid.UUID):
        uid = str(user_id)
        if uid not in user_data_events:
            return HTTPException(status_code=404, detail="User not found")
        last_time = time.time()
        try:
            while True:
                params = await websocket.receive_json()
                params = pipeline.InputParams(**params)
                params = SimpleNamespace(**params.dict())
                if hasattr(params, "image"):
                    image_data = await websocket.receive_bytes()
                    pil_image = Image.open(io.BytesIO(image_data))
                    params.image = pil_image
                user_data_events[uid].update_data({"params": params})
                if args.timeout > 0 and time.time() - last_time > args.timeout:
                    await websocket.send_json(
                        {
                            "status": "timeout",
                            "message": "Your session has ended",
                            "userId": uid,
                        }
                    )
                    await websocket.close()
                    return

        except Exception as e:
            logging.error(f"Error: {e}")
            traceback.print_exc()

    # route to setup frontend
    @app.get("/settings")
    async def settings():
        info = pipeline.Info.schema()
        input_params = pipeline.InputParams.schema()
        return JSONResponse({"info": info, "input_params": input_params})

    app.mount("/", StaticFiles(directory="public", html=True), name="public")