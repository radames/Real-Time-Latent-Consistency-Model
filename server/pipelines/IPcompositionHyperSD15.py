from diffusers import (
    DiffusionPipeline,
    TCDScheduler,
)
from compel import Compel
import torch
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image

model_id = "runwayml/stable-diffusion-v1-5"
ip_adapter_model = "ostris/ip-composition-adapter"
file_name = "ip_plus_composition_sd15.safetensors"

default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Hyper-SD Unified + IP Adpater Composition</h1>
<h3 class="text-xl font-bold">Image-to-Image ControlNet</h3>

"""


class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+SDXL+Turbo"
        title: str = "SDXL Turbo + Controlnet"
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
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            hide=True,
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            2, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.0,
            min=0,
            max=10,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        ip_adapter_scale: float = Field(
            0.8,
            min=0.0,
            max=1.0,
            step=0.001,
            title="IP Adapter Scale",
            field="range",
            hide=True,
            id="ip_adapter_scale",
        )
        eta: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Eta",
            field="range",
            hide=True,
            id="eta",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(device)

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            torch_dtype=torch_dtype,
            image_encoder=image_encoder,
            variant="fp16",
        )

        self.pipe.load_ip_adapter(
            ip_adapter_model,
            subfolder="",
            weight_name=[file_name],
            image_encoder_folder=None,
        )

        self.pipe.load_lora_weights(
            hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors")
        )
        self.pipe.fuse_lora()

        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_ip_adapter_scale([0.8])

        #         if args.compile:
        # pipe.unet = oneflow_compile(pipe.unet, options=compile_options)
        # pipe.vae.decoder = oneflow_compile(pipe.vae.decoder, options=compile_options)

        if args.sfast:
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            config = CompilationConfig.Default()
            # config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            # cofig.
            self.pipe = compile(self.pipe, config=config)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

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
                image=[Image.new("RGB", (768, 768))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        self.pipe.set_ip_adapter_scale([params.ip_adapter_scale])

        prompt_embeds = None
        prompt = params.prompt
        if hasattr(self, "compel_proc"):
            prompt_embeds = self.compel_proc(prompt)
            prompt = None

        steps = params.steps

        results = self.pipe(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            eta=params.eta,
            height=params.height,
            ip_adapter_image=[params.image],
            output_type="pil",
        )

        return results.images[0]
