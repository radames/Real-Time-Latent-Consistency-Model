from diffusers import DiffusionPipeline, AutoencoderTiny
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

base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"

page_content = """<h1 class="text-3xl font-bold">Real-Time Latent Consistency Model</h1>
<h3 class="text-xl font-bold">Text-to-Image</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/lcm#performing-inference-with-lcm"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Diffusers</a> with a MJPEG stream server
</p>
<p class="text-sm text-gray-500">
    Change the prompt to generate different images, accepts <a
    href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
</p>"""


class Pipeline:
    class Info(BaseModel):
        name: str = "txt2img"
        title: str = "Text-to-Image LCM"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "text"
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
            4, min=2, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            768, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            768, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            8.0,
            min=1,
            max=30,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        if args.oneflow_compile:
            from onediff.infer_compiler import oneflow_compile

        if args.safety_checker:
            self.pipe = DiffusionPipeline.from_pretrained(base_model)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                base_model, safety_checker=None
            )
        if args.use_taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps" and not args.oneflow_compile:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # check if computer has less than 64GB of RAM using sys or os
        if psutil.virtual_memory().total < 64 * 1024**3 and not args.oneflow_compile:
            self.pipe.enable_attention_slicing()

        if args.torch_compile:
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe(prompt="warmup", num_inference_steps=1, guidance_scale=8.0)

        if args.oneflow_compile:
            self.pipe.unet = oneflow_compile(self.pipe.unet)
            self.pipe.vae = oneflow_compile(self.pipe.vae)
            self.pipe(prompt="warmup", num_inference_steps=1, guidance_scale=8.0)

        self.compel_proc = Compel(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            truncate_long_prompts=False,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        prompt_embeds = self.compel_proc(params.prompt)
        results = self.pipe(
            prompt_embeds=prompt_embeds,
            generator=generator,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
        )
        nsfw_content_detected = (
            results.nsfw_content_detected[0]
            if "nsfw_content_detected" in results
            else False
        )
        if nsfw_content_detected:
            return None
        return results.images[0]
