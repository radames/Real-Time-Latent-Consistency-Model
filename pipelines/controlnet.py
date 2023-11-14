from diffusers import DiffusionPipeline, AutoencoderTiny
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet

from compel import Compel
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel
from PIL import Image
from typing import Callable

base_model = "SimianLuo/LCM_Dreamshaper_v7"
WIDTH = 512
HEIGHT = 512


class Pipeline:
    class InputParams(BaseModel):
        seed: int = 2159232
        prompt: str
        guidance_scale: float = 8.0
        strength: float = 0.5
        steps: int = 4
        lcm_steps: int = 50
        width: int = WIDTH
        height: int = HEIGHT

    @staticmethod
    def create_pipeline(
        args: Args, device: torch.device, torch_dtype: torch.dtype
    ) -> Callable[["Pipeline.InputParams"], Image.Image]:
        if args.safety_checker:
            pipe = DiffusionPipeline.from_pretrained(base_model)
        else:
            pipe = DiffusionPipeline.from_pretrained(base_model, safety_checker=None)
        if args.use_taesd:
            pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", torch_dtype=torch_dtype, use_safetensors=True
            )

        pipe.set_progress_bar_config(disable=True)
        pipe.to(device=device, dtype=torch_dtype)
        pipe.unet.to(memory_format=torch.channels_last)

        # check if computer has less than 64GB of RAM using sys or os
        if psutil.virtual_memory().total < 64 * 1024**3:
            pipe.enable_attention_slicing()

        if args.torch_compile:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

            pipe(prompt="warmup", num_inference_steps=1, guidance_scale=8.0)

        compel_proc = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
        )

        def predict(params: "Pipeline.InputParams") -> Image.Image:
            generator = torch.manual_seed(params.seed)
            prompt_embeds = compel_proc(params.prompt)
            # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
            results = pipe(
                prompt_embeds=prompt_embeds,
                generator=generator,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                width=params.width,
                height=params.height,
                original_inference_steps=params.lcm_steps,
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

        return predict
