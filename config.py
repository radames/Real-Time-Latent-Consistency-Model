from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    reload: bool
    mode: str
    max_queue_size: int
    timeout: float
    safety_checker: bool
    torch_compile: bool
    oneflow_compile: bool
    use_taesd: bool
    pipeline: str
    ssl_certfile: str
    ssl_keyfile: str


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None) == "True"
ONEFLOW_COMPILE = os.environ.get("ONEFLOW_COMPILE", None) == "True"
USE_TAESD = os.environ.get("USE_TAESD", None) == "True"
default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))
default_mode = os.getenv("MODE", "default")

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--mode", type=str, default=default_mode, help="App Inferece Mode: txt2img, img2img"
)
parser.add_argument(
    "--max-queue-size",
    "--max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    "--safety_checker",
    type=bool,
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--torch-compile",
    "--torch_compile",
    type=bool,
    default=TORCH_COMPILE,
    help="Torch Compile",
)
parser.add_argument(
    "--oneflow-compile",
    "--oneflow_compile",
    type=bool,
    default=ONEFLOW_COMPILE,
    help="User Oneflow compile https://github.com/Oneflow-Inc/oneflow",
)
parser.add_argument(
    "--use-taesd",
    "--use_taesd",
    type=bool,
    default=USE_TAESD,
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--pipeline",
    type=str,
    default="txt2img",
    help="Pipeline to use",
)
parser.add_argument(
    "--ssl-certfile",
    "--ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    "--ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)

args = Args(**vars(parser.parse_args()))
