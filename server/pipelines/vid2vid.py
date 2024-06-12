from .utils.v2v_wrapper import StreamV2VWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image

base_model = "runwayml/stable-diffusion-v1-5"

default_prompt = "A man is talking"

page_content = """<h1 class="text-3xl font-bold">StreamV2V by <a
    href="https://jeff-liangf.github.io/projects/streamv2v/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Jeff-LiangF
</a></h1>
<h2>Duplicate this space for fast and private usage - thank you!</h2>
<p class="text-sm">
    This demo showcases
    <a
    href="https://jeff-liangf.github.io/projects/streamv2v/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamV2V
</a>
video-to-video pipeline using
    <a
    href="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">4-step LCM LORA</a
    > with a MJPEG stream server.
</p>
<p class="text-sm">
The base model is <a
href="https://huggingface.co/runwayml/stable-diffusion-v1-5"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD 1.5</a
    >. We also build in <a
    href="https://github.com/Jeff-LiangF/streamv2v/tree/main/demo_w_camera#download-lora-weights-for-better-stylization"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">some LORAs
</a> for better stylization.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()

        self.stream = StreamV2VWrapper(
            model_id_or_path=base_model,
            t_index_list=[30, 35, 40, 45],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            warmup=10,
            acceleration="sfast",
            do_add_noise=True,
            output_type="pil",
            use_denoising_batch=True,
            use_cached_attn=True,
            use_feature_injection=True,
            feature_injection_strength=0.8,
            feature_similarity_threshold=0.98,
            cache_interval=4,
            cache_maxframes=1,
            use_tome_cache=True,
            seed=1,
        )
        self._init_lora()
        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            num_inference_steps=50,
            guidance_scale=1.0,
        )

        self.lora_active = False
        self.lora_trigger_words = [
            "pixelart",
            "pixel art",
            "Pixel art",
            "PixArFK" "lowpoly",
            "low poly",
            "Low poly",
            "Claymation",
            "claymation",
            "crayons",
            "Crayons",
            "crayons doodle",
            "Crayons doodle",
            "sketch",
            "Sketch",
            "pencil drawing",
            "Pencil drawing",
            "oil painting",
            "Oil painting",
        ]

    def _init_lora(self):
        self.stream.stream.load_lora(
            "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
            weight_name="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
            adapter_name="pixelart",
        )
        self.stream.stream.load_lora(
            "./lora_weights/low_poly.safetensors", adapter_name="lowpoly"
        )
        self.stream.stream.load_lora(
            "./lora_weights/Claymation.safetensors", adapter_name="claymation"
        )
        self.stream.stream.load_lora(
            "./lora_weights/doodle.safetensors", adapter_name="crayons"
        )
        self.stream.stream.load_lora(
            "./lora_weights/Sketch_offcolor.safetensors", adapter_name="sketch"
        )
        self.stream.stream.load_lora(
            "./lora_weights/bichu-v0612.safetensors", adapter_name="oilpainting"
        )

    def _activate_lora(self, prompt: str):
        if any(
            word in prompt for word in ["pixelart", "pixel art", "Pixel art", "PixArFK"]
        ):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "pixelart"], adapter_weights=[1.0, 1.0]
            )
            print(
                "Use LORA: pixelart in ./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
            )
        elif any(word in prompt for word in ["lowpoly", "low poly", "Low poly"]):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "lowpoly"], adapter_weights=[1.0, 1.0]
            )
            print("Use LORA: lowpoly in ./lora_weights/low_poly.safetensors")
        elif any(word in prompt for word in ["Claymation", "claymation"]):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "claymation"], adapter_weights=[1.0, 1.0]
            )
            print("Use LORA: claymation in ./lora_weights/Claymation.safetensors")
        elif any(
            word in prompt
            for word in ["crayons", "Crayons", "crayons doodle", "Crayons doodle"]
        ):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "crayons"], adapter_weights=[1.0, 1.0]
            )
            print("Use LORA: crayons in ./lora_weights/doodle.safetensors")
        elif any(
            word in prompt
            for word in ["sketch", "Sketch", "pencil drawing", "Pencil drawing"]
        ):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "sketch"], adapter_weights=[1.0, 1.0]
            )
            print("Use LORA: sketch in ./lora_weights/Sketch_offcolor.safetensors")
        elif any(word in prompt for word in ["oil painting", "Oil painting"]):
            self.stream.stream.pipe.set_adapters(
                ["lcm", "oilpainting"], adapter_weights=[1.0, 1.0]
            )
            print("Use LORA: oilpainting in ./lora_weights/bichu-v0612.safetensors")

    def _deactivate_lora(self):
        self.stream.stream.pipe.set_adapters("lcm")
        print("Deactivate LORA, back to SD1.5")

    def _check_trigger_words(self, prompt: str):
        return any(word in prompt for word in self.lora_trigger_words)

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        if self._check_trigger_words(params.prompt):
            if not self.lora_active:
                self._activate_lora(params.prompt)
                self.lora_active = True
        else:
            if self.lora_active:
                self._deactivate_lora()
                self.lora_active = False

        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
