import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.models.attention_processor import XFormersAttnProcessor, AttnProcessor2_0
from PIL import Image

from streamv2v import StreamV2V
from streamv2v.image_utils import postprocess_image
from streamv2v.models.attention_processor import (
    CachedSTXFormersAttnProcessor,
    CachedSTAttnProcessor2_0,
)


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamV2VWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        use_cached_attn: bool = True,
        use_feature_injection: bool = True,
        feature_injection_strength: float = 0.8,
        feature_similarity_threshold: float = 0.98,
        cache_interval: int = 4,
        cache_maxframes: int = 1,
        use_tome_cache: bool = True,
        tome_metric: str = "keys",
        tome_ratio: float = 0.5,
        use_grid: bool = False,
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ):
        """
        Initializes the StreamV2VWrapper.
        Parameters
        ----------
        model_id_or_path : str
            The model identifier or path to load.
        t_index_list : List[int]
            The list of indices to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            Dictionary of LoRA names and their corresponding scales,
            by default None. Example: {'LoRA_1': 0.5, 'LoRA_2': 0.7, ...}
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The type of output image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The identifier for the LCM-LoRA to load, by default None.
            If None, the default LCM-LoRA ("latent-consistency/lcm-lora-sdv1-5") is used.
        vae_id : Optional[str], optional
            The identifier for the VAE to load, by default None.
            If None, the default TinyVAE ("madebyollin/taesd") is used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The data type for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The size of the frame buffer for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "xformers".
        do_add_noise : bool, optional
            Whether to add noise during denoising steps, by default True.
        device_ids : Optional[List[int]], optional
            List of device IDs to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filtering, by default False.
        similar_image_filter_threshold : float, optional
            The threshold for the similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The maximum number of frames to skip for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"], optional
            The CFG type for img2img mode, by default "self".
        use_cached_attn : bool, optional
            Whether to cache self-attention maps from previous frames to improve temporal consistency, by default True.
        use_feature_injection : bool, optional
            Whether to use feature maps from previous frames to improve temporal consistency, by default True.
        feature_injection_strength : float, optional
            The strength of feature injection, by default 0.8.
        feature_similarity_threshold : float, optional
            The similarity threshold for feature injection, by default 0.98.
        cache_interval : int, optional
            The interval at which to cache attention maps, by default 4.
        cache_maxframes : int, optional
            The maximum number of frames to cache attention maps, by default 1.
        use_tome_cache : bool, optional
            Whether to use Tome caching, by default True.
        tome_metric : str, optional
            The metric to use for Tome, by default "keys".
        tome_ratio : float, optional
            The ratio for Tome, by default 0.5.
        use_grid : bool, optional
            Whether to use grid, by default False.
        seed : int, optional
            The seed for random number generation, by default 2.
        use_safety_checker : bool, optional
            Whether to use a safety checker, by default False.
        engine_dir : Optional[Union[str, Path]], optional
            The directory for the engine, by default "engines".
        """
        # TODO: Test SD turbo
        self.sd_turbo = "turbo" in model_id_or_path

        assert use_denoising_batch, "vid2vid mode must use denoising batch for now."

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_cached_attn = use_cached_attn
        self.use_feature_injection = use_feature_injection
        self.feature_injection_strength = feature_injection_strength
        self.feature_similarity_threshold = feature_similarity_threshold
        self.cache_interval = cache_interval
        self.cache_maxframes = cache_maxframes
        self.use_tome_cache = use_tome_cache
        self.tome_metric = tome_metric
        self.tome_ratio = tome_ratio
        self.use_grid = use_grid
        self.use_safety_checker = use_safety_checker

        self.stream: StreamV2V = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.
        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img
        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.
        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        return self.img2img(image, prompt)

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.
        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.
        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.
        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.
        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.
        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.
        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamV2V:
        """
                Loads the model.
                This method does the following:
                1. Loads the model from the model_id_or_path.
                2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
                3. Loads the VAE model from the vae_id if needed.
                4. Enables acceleration if needed.
                5. Prepares the model for inference.
                6. Load the safety checker if needed.
                Parameters
                ----------
                model_id_or_path : str
                    The model id or path to load.
                t_index_list : List[int]
                    The t_index_list to use for inference.
                lora_dict : Optional[Dict[str, float]], optional
                    The lora_dict to load, by default None.
                    Keys are the LoRA names and values are the LoRA scales.
                    Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
                lcm_lora_id : Optional[str], optional
                    The lcm_lora_id to load, by default None.
                vae_id : Optional[str], optional
                    The vae_id to load, by default None.
                acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
                    The acceleration method, by default "tensorrt".
                warmup : int, optional
                    The number of warmup steps to perform, by default 10.
                do_add_noise : bool, optional
                    Whether to add noise for following denoising steps or not,
                    by default True.
                use_lcm_lora : bool, optional
                    Whether to use LCM-LoRA or not, by default True.
                use_tiny_vae : bool, optional
                    Whether to use TinyVAE or not, by default True.
                cfg_type : Literal["none", "full", "self", "initialize"],
                optional
                    The cfg_type for img2img mode, by default "        seed : int, optional
        ".
                seed : int, optional
                    The seed, by default 2.
                Returns
                -------
                StreamV2V
                    The loaded model.
        """

        try:  # Load from local directory
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)

        except ValueError:  # Load from huggingface
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
                model_id_or_path,
            ).to(device=self.device, dtype=self.dtype)
        except Exception:  # No model found
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamV2V(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id,
                        adapter_name="lcm",
                    )
                else:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict="latent-consistency/lcm-lora-sdv1-5",
                        adapter_name="lcm",
                    )

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
                if self.use_cached_attn:
                    attn_processors = stream.pipe.unet.attn_processors
                    new_attn_processors = {}
                    for key, attn_processor in attn_processors.items():
                        assert isinstance(
                            attn_processor, XFormersAttnProcessor
                        ), "We only replace 'XFormersAttnProcessor' to 'CachedSTXFormersAttnProcessor'"
                        new_attn_processors[key] = CachedSTXFormersAttnProcessor(
                            name=key,
                            use_feature_injection=self.use_feature_injection,
                            feature_injection_strength=self.feature_injection_strength,
                            feature_similarity_threshold=self.feature_similarity_threshold,
                            interval=self.cache_interval,
                            max_frames=self.cache_maxframes,
                            use_tome_cache=self.use_tome_cache,
                            tome_metric=self.tome_metric,
                            tome_ratio=self.tome_ratio,
                            use_grid=self.use_grid,
                        )
                    stream.pipe.unet.set_attn_processor(new_attn_processors)

            if acceleration == "tensorrt":
                if self.use_cached_attn:
                    raise NotImplementedError(
                        "TensorRT seems not support the costom attention_processor"
                    )
                else:
                    stream.pipe.enable_xformers_memory_efficient_attention()
                    if self.use_cached_attn:
                        attn_processors = stream.pipe.unet.attn_processors
                        new_attn_processors = {}
                        for key, attn_processor in attn_processors.items():
                            assert isinstance(
                                attn_processor, XFormersAttnProcessor
                            ), "We only replace 'XFormersAttnProcessor' to 'CachedSTXFormersAttnProcessor'"
                            new_attn_processors[key] = CachedSTXFormersAttnProcessor(
                                name=key,
                                use_feature_injection=self.use_feature_injection,
                                feature_injection_strength=self.feature_injection_strength,
                                feature_similarity_threshold=self.feature_similarity_threshold,
                                interval=self.cache_interval,
                                max_frames=self.cache_maxframes,
                                use_tome_cache=self.use_tome_cache,
                                tome_metric=self.tome_metric,
                                tome_ratio=self.tome_ratio,
                                use_grid=self.use_grid,
                            )
                        stream.pipe.unet.set_attn_processor(new_attn_processors)

                from polygraphy import cuda
                from streamv2v.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamv2v.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamv2v.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--cache--{self.use_cached_attn}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--cache--{self.use_cached_attn}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=stream.frame_bff_size,
                        min_batch_size=stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=stream.frame_bff_size,
                    )

                cuda_steram = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_steram, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_steram,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                if self.use_cached_attn:
                    raise NotImplementedError
                from streamv2v.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
