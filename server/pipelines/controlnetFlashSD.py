from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderTiny,
    LCMScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import torch
from pipelines.utils.canny_gpu import SobelOperator

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

controlnet_model = "lllyasviel/control_v11p_sd15_canny"
model_id = "runwayml/stable-diffusion-v1-5"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
page_content = """
<h1 class="text-3xl font-bold">Flash-SD</h1>
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
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
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
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            controlnet=controlnet_canny,
            torch_dtype=torch_dtype,
        )

        self.pipe.scheduler = LCMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            timestep_spacing="trailing",
        )

        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            )
        self.pipe.load_lora_weights("jasperai/flash-sd")
        self.pipe.fuse_lora()

        self.canny_torch = SobelOperator(device=device)

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
                control_image=[Image.new("RGB", (768, 768))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
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
        steps = params.steps
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
            num_inference_steps=steps,
            guidance_scale=0,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
        )

        result_image = results.images[0]
        if params.debug_canny:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image
