from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
)
from compel import Compel
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

taesd_model = "madebyollin/taesd"
controlnet_model = "monster-labs/control_v1p_sd15_qrcode_monster"
base_model = "nitrosocke/mo-di-diffusion"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
default_prompt = "abstract art of a men with curly hair by Pablo Picasso"
page_content = """
<h1 class="text-3xl font-bold">Real-Time Latent Consistency Model SDv1.5</h1>
<h3 class="text-xl font-bold">LCM + LoRA + Controlnet + QRCode</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/blog/lcm_lora"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM LoRA</a>
+ ControlNet + Image to Imasge pipeline using
    <a
    href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm#performing-inference-with-lcm"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Diffusers</a
    > with a MJPEG stream server.
</p>
<p class="text-sm text-gray-500">
    Change the prompt to generate different images, accepts <a
    href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+loras+sd15"
        title: str = "LCM + LoRA + Controlnet"
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
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            5, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.0,
            min=0,
            max=2,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.6,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Scale",
            field="range",
            hide=True,
            id="controlnet_scale",
        )
        controlnet_start: float = Field(
            0.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Start",
            field="range",
            hide=True,
            id="controlnet_start",
        )
        controlnet_end: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet End",
            field="range",
            hide=True,
            id="controlnet_end",
        )
        blend: float = Field(
            0.1,
            min=0.0,
            max=1.0,
            step=0.001,
            title="Blend",
            field="range",
            hide=True,
            id="blend",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        controlnet_qrcode = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype, subfolder="v2"
        ).to(device)

        if args.safety_checker:
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_model,
                controlnet=controlnet_qrcode,
            )
        else:
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_model,
                safety_checker=None,
                controlnet=controlnet_qrcode,
            )

        self.control_image = Image.open("qr-code.png").convert("RGB").resize((512, 512))

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        # Load LCM LoRA
        self.pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
        self.pipe.to(device=device, dtype=torch_dtype).to(device)
        if args.compel:
            self.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=False,
            )
        if args.torch_compile:
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )
            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (512, 512))],
                control_image=[Image.new("RGB", (512, 512))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)

        prompt = f"modern disney style {params.prompt}"
        prompt_embeds = None
        prompt = params.prompt
        if hasattr(self, "compel_proc"):
            prompt_embeds = self.compel_proc(prompt)
            prompt = None

        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        blend_qr_image = Image.blend(
            params.image, self.control_image, alpha=params.blend
        )
        results = self.pipe(
            image=blend_qr_image,
            control_image=self.control_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
        )

        return results.images[0]
