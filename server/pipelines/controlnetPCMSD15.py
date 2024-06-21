from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    TCDScheduler,
    AutoencoderTiny,
)
from compel import Compel
import torch
from pipelines.utils.canny_gpu import SobelOperator

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image

taesd_model = "madebyollin/taesd"
controlnet_model = "lllyasviel/control_v11p_sd15_canny"
base_model_id = "runwayml/stable-diffusion-v1-5"
pcm_base = "wangfuyun/PCM_Weights"
pcm_lora_ckpts = {
    "2-Step": ["pcm_sd15_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_sd15_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_sd15_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_sd15_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_sd15_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_sd15_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_sd15_normalcfg_16step_converted.safetensors", 16, 7.5],
}
default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
page_content = """

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
        lora_ckpt_id: str = Field(
            "4-Step",
            title="PCM Base Model",
            values=list(pcm_lora_ckpts.keys()),
            field="select",
            id="lora_ckpt_id",
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
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

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            safety_checker=None,
            controlnet=controlnet_canny,
        )

        self.canny_torch = SobelOperator(device=device)

        self.pipe.scheduler = TCDScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        )

        self.pipe.set_progress_bar_config(disable=True)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        self.loaded_lora = "4-Step"
        self.pipe.load_lora_weights(
            pcm_base,
            weight_name=pcm_lora_ckpts[self.loaded_lora][0],
            subfolder="sd15",
        )
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
                image=[Image.new("RGB", (768, 768))],
                control_image=[Image.new("RGB", (768, 768))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        guidance_scale = pcm_lora_ckpts[params.lora_ckpt_id][2]
        steps = pcm_lora_ckpts[params.lora_ckpt_id][1]

        if self.loaded_lora != params.lora_ckpt_id:
            checkpoint = pcm_lora_ckpts[params.lora_ckpt_id][0]
            self.pipe.load_lora_weights(
                pcm_base,
                weight_name=checkpoint,
                subfolder="sd15",
            )
            self.loaded_lora = params.lora_ckpt_id

        prompt_embeds = None
        prompt = params.prompt
        if hasattr(self, "compel_proc"):
            prompt_embeds = self.compel_proc(prompt)
            prompt = None
        control_image = self.canny_torch(
            params.image, params.canny_low_threshold, params.canny_high_threshold
        )
        strength = params.strength

        results = self.pipe(
            image=params.image,
            control_image=control_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
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
