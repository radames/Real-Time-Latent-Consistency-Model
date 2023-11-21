from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
)
from compel import Compel
import torch
from pipelines.utils.canny_gpu import SobelOperator

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image

taesd_model = "madebyollin/taesd"
controlnet_model = "lllyasviel/control_v11p_sd15_canny"
# base model with activation token, it will prepend the prompt with the activation token
base_models = {
    "plasmo/woolitize": "woolitize",
    "nitrosocke/Ghibli-Diffusion": "ghibli style",
    "nitrosocke/mo-di-diffusion": "modern disney style",
}
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"


default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"


class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+loras+sd15"
        title: str = "LCM + LoRA + Controlnet "
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        base_model_id: str = Field(
            "plasmo/woolitize",
            title="Base Model",
            values=list(base_models.keys()),
            field="select",
            id="base_model_id",
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            4, min=2, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.2,
            min=0,
            max=2,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
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
            0.8,
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
        ).to(device)

        self.pipes = {}

        if args.safety_checker:
            for base_model_id in base_models.keys():
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    base_model_id,
                    controlnet=controlnet_canny,
                )
            self.pipes[base_model_id] = pipe
        else:
            for base_model_id in base_models.keys():
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    base_model_id,
                    safety_checker=None,
                    controlnet=controlnet_canny,
                )
                self.pipes[base_model_id] = pipe

        self.canny_torch = SobelOperator(device=device)

        for pipe in self.pipes.values():
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            pipe.to(device=device, dtype=torch_dtype).to(device)
            if device.type != "mps":
                pipe.unet.to(memory_format=torch.channels_last)

            if psutil.virtual_memory().total < 64 * 1024**3:
                pipe.enable_attention_slicing()

            # Load LCM LoRA
            pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")
            pipe.compel_proc = Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                truncate_long_prompts=False,
            )
            if args.torch_compile:
                pipe.unet = torch.compile(
                    pipe.unet, mode="reduce-overhead", fullgraph=True
                )
                pipe.vae = torch.compile(
                    pipe.vae, mode="reduce-overhead", fullgraph=True
                )
                pipe(
                    prompt="warmup",
                    image=[Image.new("RGB", (768, 768))],
                    control_image=[Image.new("RGB", (768, 768))],
                )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        print(f"Using model: {params.base_model_id}")
        pipe = self.pipes[params.base_model_id]

        activation_token = base_models[params.base_model_id]
        prompt = f"{activation_token} {params.prompt}"
        prompt_embeds = pipe.compel_proc(prompt)
        control_image = self.canny_torch(
            params.image, params.canny_low_threshold, params.canny_high_threshold
        )

        results = pipe(
            image=params.image,
            control_image=control_image,
            prompt_embeds=prompt_embeds,
            generator=generator,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
        )

        nsfw_content_detected = (
            results.nsfw_content_detected[0]
            if "nsfw_content_detected" in results
            else False
        )
        if nsfw_content_detected:
            return None
        result_image = results.images[0]
        if params.debug_canny:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image
