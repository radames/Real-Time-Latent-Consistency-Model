from diffusers import (
    UNet2DConditionModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    AutoencoderTiny,
    EulerDiscreteScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import torch
from pipelines.utils.canny_gpu import SobelOperator
from pipelines.utils.safety_checker import SafetyChecker
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

controlnet_model = "diffusers/controlnet-canny-sdxl-1.0-small"
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_2step_unet.safetensors"
taesd_model = "madebyollin/taesdxl"
NUM_STEPS = 2


default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Real-Time Latent Consistency Model SDXL</h1>
<h3 class="text-xl font-bold">SDXL-Lightining + Controlnet</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/blog/lcm_lora"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">LCM LoRA</a>
+ SDXL + Controlnet + Image to Image pipeline using
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
        name: str = "controlnet+loras+sdxl+lightning"
        title: str = "SDXL + LCM + LoRA + Controlnet"
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
        width: int = Field(
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            1024, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.0,
            min=0,
            max=2.0,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            1,
            min=0.25,
            max=1.0,
            step=0.0001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            0.5,
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
        canny_low_threshold: float = Field(
            0.31,
            min=0,
            max=1.0,
            step=0.001,
            title="Canny Low Threshold",
            field="range",
            hide=True,
            id="canny_low_threshold",
        )
        canny_high_threshold: float = Field(
            0.125,
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
        self.safety_checker = None
        if args.safety_checker:
            self.safety_checker = SafetyChecker(device=device.type)

        if args.taesd:
            vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
            )

        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype
        )

        unet = UNet2DConditionModel.from_config(base, subfolder="unet")
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device.type))
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=torch_dtype,
            variant="fp16",
            controlnet=controlnet_canny,
            vae=vae,
        )

        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

        self.canny_torch = SobelOperator(device=device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)

        if args.sfast:
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.pipe = compile(self.pipe, config=config)

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
                control_image=[Image.new("RGB", (768, 768))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image | None:
        generator = torch.manual_seed(params.seed)

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

        control_image = self.canny_torch(
            params.image, params.canny_low_threshold, params.canny_high_threshold
        )
        steps = NUM_STEPS
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        results = self.pipe(
            image=params.image,
            control_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=NUM_STEPS,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
        )
        images = results.images
        if self.safety_checker:
            images, has_nsfw_concepts = self.safety_checker(images)
            if any(has_nsfw_concepts):
                return None

        result_image = images[0]
        if params.debug_canny:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image
