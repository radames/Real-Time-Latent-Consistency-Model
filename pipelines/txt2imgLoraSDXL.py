from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    AutoencoderKL,
)
from compel import Compel, ReturnedEmbeddingsType
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image

controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"


default_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"


class Pipeline:
    class Info(BaseModel):
        name: str = "LCM+Lora+SDXL"
        title: str = "Text-to-Image SDXL + LCM + LoRA"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "text"

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
            4, min=2, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            1024, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.0,
            min=0,
            max=20,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
        )
        if args.safety_checker:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                vae=vae,
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                safety_checker=None,
                vae=vae,
            )
        # Load LCM LoRA
        self.pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype).to(device)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if psutil.virtual_memory().total < 64 * 1024**3:
            self.pipe.enable_attention_slicing()

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
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)

        prompt_embeds, pooled_prompt_embeds = self.pipe.compel_proc(
            [params.prompt, params.negative_prompt]
        )
        results = self.pipe(
            prompt_embeds=prompt_embeds[0:1],
            pooled_prompt_embeds=pooled_prompt_embeds[0:1],
            negative_prompt_embeds=prompt_embeds[1:2],
            negative_pooled_prompt_embeds=pooled_prompt_embeds[1:2],
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
        result_image = results.images[0]

        return result_image
