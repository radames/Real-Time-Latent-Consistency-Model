from pydantic import BaseModel, field_validator
import argparse
import os
from typing import Annotated


class Args(BaseModel):
    host: str
    port: int
    reload: bool
    max_queue_size: int
    timeout: float
    safety_checker: bool
    torch_compile: bool
    taesd: bool
    pipeline: str
    ssl_certfile: str | None
    ssl_keyfile: str | None
    compel: bool = False
    debug: bool = False
    pruna: bool = False

    def pretty_print(self) -> None:
        print("\n")
        for field, value in self.model_dump().items():
            print(f"{field}: {value}")
        print("\n")

    @field_validator("ssl_keyfile")
    @classmethod
    def validate_ssl_keyfile(cls, v: str | None, info) -> str | None:
        """Validate that if ssl_certfile is provided, ssl_keyfile is also provided."""
        ssl_certfile = info.data.get("ssl_certfile")
        if ssl_certfile and not v:
            raise ValueError(
                "If ssl_certfile is provided, ssl_keyfile must also be provided"
            )
        return v


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None) == "True"
USE_TAESD = os.environ.get("USE_TAESD", "False") == "True"
default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    dest="safety_checker",
    action="store_true",
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--torch-compile",
    dest="torch_compile",
    action="store_true",
    default=TORCH_COMPILE,
    help="Torch Compile",
)
parser.add_argument(
    "--taesd",
    dest="taesd",
    action="store_true",
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
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Debug",
)
parser.add_argument(
    "--compel",
    action="store_true",
    default=False,
    help="Compel",
)
parser.add_argument(
    "--pruna",
    action="store_true",
    default=False,
    help="Enable Pruna",
)
parser.set_defaults(taesd=USE_TAESD)

config = Args.model_validate(vars(parser.parse_args()))
config.pretty_print()
