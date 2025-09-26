from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import markdown2
from PIL import Image
import logging
from config import config, Args
from connection_manager import ConnectionManager, ServerFullException
from uuid import UUID
import time
from typing import Any, Protocol, runtime_checkable, AsyncGenerator
from util import pil_to_frame, bytes_to_pil, is_firefox, get_pipeline_class, ParamsModel
from device import device, torch_dtype
import asyncio
import os
import time

# Common WebSocket error messages that indicate disconnection
ERROR_MESSAGES = [
    "WebSocket is not connected",
    'Cannot call "send" once a close message has been sent',
    'Cannot call "receive" once a close message has been sent',
    "WebSocket is disconnected",
]


@runtime_checkable
class BasePipeline(Protocol):
    class Info:
        @classmethod
        def model_json_schema(cls, **kwargs) -> dict[str, Any]: ...

        page_content: str | None
        input_mode: str

    class InputParams(ParamsModel):
        @classmethod
        def model_json_schema(cls, **kwargs) -> dict[str, Any]: ...

    def predict(self, params: ParamsModel) -> Image.Image | None: ...


THROTTLE = 1.0 / 120


class App:
    def __init__(self, config: Args, pipeline_instance: BasePipeline):
        self.args = config
        self.pipeline = pipeline_instance
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.safety_checker = None
        if self.args.safety_checker:
            from pipelines.utils.safety_checker import SafetyChecker

            self.safety_checker = SafetyChecker(device=device.type)
        self.init_app()

    def init_app(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: UUID, websocket: WebSocket) -> None:
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            except WebSocketDisconnect as disconnect_error:
                # Handle websocket disconnection event
                code = disconnect_error.code
                if code == 1006:  # ABNORMAL_CLOSURE
                    logging.info(
                        f"WebSocket abnormally closed for user {user_id}: Connection was closed without a proper close handshake"
                    )
                else:
                    logging.info(
                        f"WebSocket disconnected for user {user_id} with code {code}: {disconnect_error.reason}"
                    )
            except RuntimeError as e:
                if any(err in str(e) for err in ERROR_MESSAGES):
                    logging.info(f"WebSocket disconnected for user {user_id}: {e}")
                else:
                    logging.error(f"Runtime error in websocket endpoint: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in websocket endpoint: {e}")
            finally:
                # Always ensure we disconnect the user
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: UUID) -> None:
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if data is None:
                        # Check if the user is still connected - they might have disconnected
                        if not self.conn_manager.check_user(user_id):
                            logging.info(
                                f"User {user_id} disconnected, exiting handle_websocket_data loop"
                            )
                            return
                        continue

                    # Validate that data is a dictionary and has a status field
                    if not isinstance(data, dict) or "status" not in data:
                        logging.error(
                            f"Invalid data format received from user {user_id}: {data}"
                        )
                        continue

                    if data["status"] == "next_frame":
                        info = self.pipeline.Info()
                        params_data = await self.conn_manager.receive_json(user_id)
                        if params_data is None:
                            # Check if the user is still connected
                            if not self.conn_manager.check_user(user_id):
                                logging.info(
                                    f"User {user_id} disconnected during params reception"
                                )
                                return
                            continue

                        params = self.pipeline.InputParams.model_validate(params_data)

                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if image_data is None:
                                # Check if the user is still connected
                                if not self.conn_manager.check_user(user_id):
                                    logging.info(
                                        f"User {user_id} disconnected during image reception"
                                    )
                                    return
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue

                            # Add the image directly to the model using setattr
                            # This works because we've configured the ParamsModel to allow extra fields
                            setattr(params, "image", bytes_to_pil(image_data))

                        await self.conn_manager.update_data(user_id, params)
                        await self.conn_manager.send_json(user_id, {"status": "wait"})

            except RuntimeError as e:
                error_msg = str(e)
                if any(err in error_msg for err in ERROR_MESSAGES):
                    logging.info(
                        f"WebSocket disconnected for user {user_id}: {error_msg}"
                    )
                else:
                    logging.error(f"Websocket Runtime Error: {e}, {user_id}")
                # Ensure disconnect is called
                await self.conn_manager.disconnect(user_id)
            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id}")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size() -> JSONResponse:
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: UUID, request: Request) -> StreamingResponse:
            try:

                async def generate() -> AsyncGenerator[bytes, None]:
                    last_params: ParamsModel | None = None
                    while True:
                        # Check if the user is still connected
                        if not self.conn_manager.check_user(user_id):
                            logging.info(f"User {user_id} disconnected from stream")
                            break

                        last_time = time.time()
                        try:
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                        except Exception as e:
                            logging.error(f"Error sending to websocket in stream: {e}")
                            # User might have disconnected
                            if not self.conn_manager.check_user(user_id):
                                logging.info(f"User {user_id} disconnected from stream")
                                break
                            await asyncio.sleep(THROTTLE)
                            continue

                        params = await self.conn_manager.get_latest_data(user_id)

                        if params is None:
                            await asyncio.sleep(THROTTLE)
                            continue

                        try:
                            # Check if the params haven't changed since last time
                            if (
                                last_params is not None
                                and params.model_dump() == last_params.model_dump()
                            ):
                                await asyncio.sleep(THROTTLE)
                                continue

                            last_params = params
                            image = self.pipeline.predict(params)
                        except Exception as e:
                            logging.error(
                                f"Error processing params for user {user_id}: {e}"
                            )
                            await asyncio.sleep(THROTTLE)
                            continue

                        if (
                            self.args.safety_checker
                            and self.safety_checker is not None
                            and image is not None
                        ):
                            image, has_nsfw_concept = self.safety_checker(image)
                            if has_nsfw_concept:
                                image = None

                        if image is None:
                            continue
                        frame = pil_to_frame(image)
                        yield frame
                        # https://bugs.chromium.org/p/chromium/issues/detail?id=1250396
                        if not is_firefox(request.headers["user-agent"]):
                            yield frame
                        if self.args.debug:
                            print(f"Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except WebSocketDisconnect as disconnect_error:
                # Handle websocket disconnection event
                code = disconnect_error.code
                if code == 1006:  # ABNORMAL_CLOSURE
                    logging.info(
                        f"WebSocket abnormally closed during streaming for user {user_id}: Connection was closed without a proper close handshake"
                    )
                else:
                    logging.info(
                        f"WebSocket disconnected during streaming for user {user_id} with code {code}: {disconnect_error.reason}"
                    )

                # Clean disconnection without error response
                await self.conn_manager.disconnect(user_id)
                raise HTTPException(status_code=204, detail="Connection closed")
            except RuntimeError as e:
                error_msg = str(e)
                if any(err in error_msg for err in ERROR_MESSAGES):
                    logging.info(
                        f"WebSocket disconnected during streaming for user {user_id}: {error_msg}"
                    )
                    # Clean disconnection without error response
                    await self.conn_manager.disconnect(user_id)
                    raise HTTPException(status_code=204, detail="Connection closed")
                else:
                    logging.error(f"Streaming Runtime Error: {e}, {user_id}")
                    raise HTTPException(status_code=500, detail="Streaming error")
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id}")
                # Always ensure we disconnect the user on error
                await self.conn_manager.disconnect(user_id)
                raise HTTPException(status_code=500, detail="Streaming error")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings() -> JSONResponse:
            info_schema = self.pipeline.Info.model_json_schema()
            info = self.pipeline.Info()
            page_content = ""
            if hasattr(info, "page_content") and info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = self.pipeline.InputParams.model_json_schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content,
                }
            )

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount(
            "/", StaticFiles(directory="frontend/public", html=True), name="public"
        )


# def create_app(config):
#     print(f"Device: {device}")
#     print(f"torch_dtype: {torch_dtype}")

#     # Create pipeline once
#     pipeline_class = get_pipeline_class(config.pipeline)
#     pipeline_instance = pipeline_class(config, device, torch_dtype)

#     # Pass the existing pipeline instance to App
#     app = App(config, pipeline_instance).app
#     return app


# Create app instance at module level
print(f"Device: {device}")
print(f"torch_dtype: {torch_dtype}")

pipeline_class = get_pipeline_class(config.pipeline)
pipeline_instance = pipeline_class(config, device, torch_dtype)
app = App(config, pipeline_instance).app  # This creates the FastAPI app instance


if __name__ == "__main__":
    import uvicorn

    # app = create_app(config)  # Create the app once

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
