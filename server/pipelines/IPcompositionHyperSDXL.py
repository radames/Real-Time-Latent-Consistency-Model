from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    TCDScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
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

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
taesd_model = "madebyollin/taesdxl"
ip_adapter_model = "ostris/ip-composition-adapter"
file_name = "ip_plus_composition_sdxl.safetensors"

default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Hyper-SDXL Unified + IP Adpater Composition</h1>
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
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            1024, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
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
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(device)

        if args.safety_checker:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                # vae=vae,
                torch_dtype=torch_dtype,
                image_encoder=image_encoder,
                variant="fp16",
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                safety_checker=None,
                torch_dtype=torch_dtype,
                vae=vae,
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
            hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-lora.safetensors")
        )
        self.pipe.fuse_lora()

        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_ip_adapter_scale([0.8])

        if args.sfast:
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            config = CompilationConfig.Default()
            # config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.pipe = compile(self.pipe, config=config)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.compel:
            self.pipe.compel_proc = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
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

        prompt = params.prompt
        negative_prompt = params.negative_prompt
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if hasattr(self.pipe, "compel_proc"):
            _prompt_embeds, pooled_prompt_embeds = self.pipe.compel_proc(
                [params.prompt, params.negative_prompt]
            )
            prompt = None
            negative_prompt = None
            prompt_embeds = _prompt_embeds[0:1]
            pooled_prompt_embeds = pooled_prompt_embeds[0:1]
            negative_prompt_embeds = _prompt_embeds[1:2]
            negative_pooled_prompt_embeds = pooled_prompt_embeds[1:2]

        steps = params.steps

        results = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            eta=params.eta,
            height=params.height,
            ip_adapter_image=[params.image],
            output_type="pil",
        )

        nsfw_content_detected = (
            results.nsfw_content_detected[0]
            if "nsfw_content_detected" in results
            else False
        )
        if nsfw_content_detected:
            return None
        result_image = results.images[0]

        return result_image
