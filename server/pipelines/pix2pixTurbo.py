import torch
from torchvision import transforms

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from pipelines.pix2pix.pix2pix_turbo import Pix2Pix_Turbo
from pipelines.utils.canny_gpu import ScharrOperator

default_prompt = "close-up photo of the joker"
page_content = """
<h1 class="text-3xl font-bold">Real-Time pix2pix_turbo</h1>
<h3 class="text-xl font-bold">pix2pix turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/GaParmar/img2img-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">One-Step Image Translation with Text-to-Image Models
    </a>
</p>
<p class="text-sm text-gray-500">
    Web app <a href="https://github.com/radames/Real-Time-Latent-Consistency-Model" target="_blank" class="text-blue-500 underline hover:no-underline">
    Real-Time Latent Consistency Models
    </a>
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image SDXL"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )

        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        noise_r: float = Field(
            1.0,
            min=0.01,
            max=3.0,
            step=0.001,
            title="Noise R",
            field="range",
            hide=True,
            id="noise_r",
        )

        deterministic: bool = Field(
            True,
            hide=True,
            title="Deterministic",
            field="checkbox",
            id="deterministic",
        )
        canny_low_threshold: float = Field(
            0.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Canny Low Threshold",
            field="range",
            hide=True,
            id="canny_low_threshold",
        )
        canny_high_threshold: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Canny High Threshold",
            field="range",
            hide=True,
            id="canny_high_threshold",
        )
        debug_canny: bool = Field(
            False,
            title="Debug Canny",
            field="checkbox",
            hide=True,
            id="debug_canny",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        self.model = Pix2Pix_Turbo("edge_to_image")
        self.canny_torch = ScharrOperator(device=device)
        self.device = device
        self.last_time = 0.0

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        canny_pil, canny_tensor = self.canny_torch(
            params.image,
            params.canny_low_threshold,
            params.canny_high_threshold,
            output_type="pil,tensor",
        )
        torch.manual_seed(params.seed)
        noise = torch.randn(
            (1, 4, params.width // 8, params.height // 8), device=self.device
        )
        canny_tensor = torch.cat((canny_tensor, canny_tensor, canny_tensor), dim=1)
        output_image = self.model(
            canny_tensor,
            params.prompt,
            params.deterministic,
            params.noise_r,
            noise,
        )
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

        result_image = output_pil
        if params.debug_canny:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = canny_pil.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))
        return result_image
