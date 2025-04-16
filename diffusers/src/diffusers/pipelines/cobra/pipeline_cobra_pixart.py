# Copyright 2024 PixArt-Alpha Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('torch_npu not found')
import html
import inspect
import re
import numpy as np
from PIL import Image
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
from transformers import T5EncoderModel, T5Tokenizer

from ...image_processor import PixArtImageProcessor
from ...models import AutoencoderKL, CausalSparseDiTModel, CausalSparseDiTControlModel
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from ...image_processor import PipelineImageInput, VaeImageProcessor
from PIL import Image
from torchvision import transforms
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixArtAlphaPipeline

        >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
        >>> pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A small cactus with a happy face in the Sahara desert."
        >>> image = pipe(prompt).images[0]
        ```
"""

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}

ASPECT_RATIO_256_BIN = {
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0],
}

def mask_to_tensor(PIL_image, normalize=True):
    """
    将图像转换为形状为 (1, 1, H, W) 的 PyTorch 张量。

    参数:
        image_path (str): 图像文件路径。
        normalize (bool): 是否将像素值归一化到 [0, 1]。默认为 True。

    返回:
        torch.Tensor: 形状为 (1, 1, H, W) 的张量。
    """
    # 打开图像并转换为灰度图像
    image = PIL_image.convert('L')

    # 将图像转换为 NumPy 数组
    image_array = np.array(image)

    # 调整数组形状为 (1, 1, H, W)
    image_array = image_array[np.newaxis, np.newaxis, :, :]

    # 将 NumPy 数组转换为 PyTorch 张量
    image_tensor = torch.from_numpy(image_array)

    # 如果需要将像素值归一化到 [0, 1]，可以除以 255
    if normalize:
        image_tensor = image_tensor.float() / 255.0

    return image_tensor

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CobraPixArtAlphaPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CausalDiT`]):
            A text conditioned `CausalDiT` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        transformer: CausalSparseDiTModel,
        controlnet: CausalSparseDiTControlModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae, transformer=transformer, scheduler=scheduler, controlnet = controlnet
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def check_inputs(
        self,
        height,
        width,
        callback_steps,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # print('generator', generator)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents_ref(self, batch_size, num_refs, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_refs,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # print('generator', generator)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        max_sequence_length: int = 120,
        cond_input: PipelineImageInput = None,
        cond_refs: list = None,
        hint_mask: PipelineImageInput = None,
        hint_color: PipelineImageInput = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        # 1. Check inputs. Raise error if not correct
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # if use_resolution_binning:
        #     if self.transformer.config.sample_size == 128:
        #         aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        #     elif self.transformer.config.sample_size == 64:
        #         aspect_ratio_bin = ASPECT_RATIO_512_BIN
        #     elif self.transformer.config.sample_size == 32:
        #         aspect_ratio_bin = ASPECT_RATIO_256_BIN
        #     else:
        #         raise ValueError("Invalid sample size")
        #     orig_height, orig_width = height, width
        #     # height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)
        #     height,width = orig_height,orig_width
        width, height = cond_input.size
        hint_width, hint_height = hint_color.size
        if hint_width != width or hint_height != height:
            raise ValueError(f"Width and height of hint_color must be the same as cond_input, but got {hint_width} and {hint_height} for cond_input with size {width} and {height}.")
        for tmp_i in range(len(cond_refs)):
            if cond_refs[tmp_i]!=[]:
                width_ref, height_ref = cond_refs[tmp_i][0].size
                break
        if width_ref*2 != width or height_ref*2 != height:
            raise ValueError(f"Width and height of cond_refs must be twice the size of cond_input, but got {width_ref} and {height_ref} for cond_input with size {width} and {height}.")

        self.check_inputs(
            height,
            width,
            callback_steps,
        )

        # 2. Default height and width to transformer
        batch_size = 1
        n_ref_idx0 = len(cond_refs[0])
        n_ref_idx1 = len(cond_refs[1])
        n_ref_idx2 = len(cond_refs[2])
        n_ref_idx3 = len(cond_refs[3])
        cond_refs_idx0 = cond_refs[0]
        cond_refs_idx1 = cond_refs[1]
        cond_refs_idx2 = cond_refs[2]
        cond_refs_idx3 = cond_refs[3]
        N_ref = n_ref_idx0 + n_ref_idx1 + n_ref_idx2 + n_ref_idx3
        num_ref_list = [n_ref_idx0, n_ref_idx1, n_ref_idx2, n_ref_idx3]
        print('num_ref_list',num_ref_list)
        num_images_per_prompt = 1

        device = self._execution_device

        prompt_embeds = torch.load('./prompt_tensor/prompt_embeds.pt').unsqueeze(0).repeat(batch_size * num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.to(dtype=self.transformer.dtype, device=device)
        prompt_attention_mask = torch.load('./prompt_tensor/prompt_attention_mask.pt').unsqueeze(0).repeat(batch_size * num_images_per_prompt,1)
        prompt_attention_mask = prompt_attention_mask.to(dtype=self.transformer.dtype, device=device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = False

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            self.transformer.dtype,
            device,
            generator,
            # latents,
        )
        latents_ref = self.prepare_latents_ref(
            batch_size * num_images_per_prompt,
            N_ref,
            latent_channels,
            height_ref,
            width_ref,
            self.transformer.dtype,
            device,
            generator,
        )


        cond_input = self.prepare_image(
            image=cond_input,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        hint_color = self.prepare_image(
            image=hint_color,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        cond_refs_idx0 = [self.prepare_image(
            image=cond_ref,
            width=width_ref,
            height=height_ref,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) for cond_ref in cond_refs_idx0]
        if len(cond_refs_idx0) > 0: 
            cond_refs_idx0 = torch.cat(cond_refs_idx0, dim=0)
        else:
            cond_refs_idx0 = torch.zeros(0,0,0,0).to(dtype=self.controlnet.dtype, device=device)
        cond_refs_idx1 = [self.prepare_image(
            image=cond_ref,
            width=width_ref,
            height=height_ref,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) for cond_ref in cond_refs_idx1]
        if len(cond_refs_idx1) > 0:
            cond_refs_idx1 = torch.cat(cond_refs_idx1, dim=0)
        else:
            cond_refs_idx1 = torch.zeros(0,0,0,0).to(dtype=self.controlnet.dtype, device=device)
        cond_refs_idx2 = [self.prepare_image(
            image=cond_ref,
            width=width_ref,
            height=height_ref,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) for cond_ref in cond_refs_idx2]
        if len(cond_refs_idx2) > 0:
            cond_refs_idx2 = torch.cat(cond_refs_idx2, dim=0)
        else:
            cond_refs_idx2 = torch.zeros(0,0,0,0).to(dtype=self.controlnet.dtype, device=device)
        cond_refs_idx3 = [self.prepare_image(
            image=cond_ref,
            width=width_ref,
            height=height_ref,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        ) for cond_ref in cond_refs_idx3]
        if len(cond_refs_idx3) > 0:
            cond_refs_idx3 = torch.cat(cond_refs_idx3, dim=0)
        else:
            cond_refs_idx3 = torch.zeros(0,0,0,0).to(dtype=self.controlnet.dtype, device=device)

        all_cond_refs = []
        if num_ref_list[0] != 0:
            all_cond_refs.append(cond_refs_idx0)
        if num_ref_list[1] != 0:
            all_cond_refs.append(cond_refs_idx1)
        if num_ref_list[2] != 0:
            all_cond_refs.append(cond_refs_idx2)
        if num_ref_list[3] != 0:
            all_cond_refs.append(cond_refs_idx3)
        cond_refs = torch.cat(all_cond_refs, dim=0)
        # print('cond_refs',cond_refs.shape)

        hint_mask = mask_to_tensor(hint_mask).to(dtype=self.controlnet.dtype, device=device)
        # print('pipeline hint_mask',torch.max(hint_mask),torch.min(hint_mask))
        # hint_mask_pil = transforms.ToPILImage()(hint_mask[0,:,:,:])
        # hint_mask_pil.save('hint_mask.png')

        



        height, width = cond_input.shape[-2:]
        # print(self.vae.dtype, self.controlnet.dtype, cond_image.dtype)
        cond_input_latent = self.vae.encode(cond_input.to(dtype = self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        cond_refs_latent=self.vae.encode(cond_refs.to(dtype = self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        hint_color_latent = self.vae.encode(hint_color.to(dtype = self.vae.dtype)).latent_dist.sample() * self.vae.config.scaling_factor


        cond_refs_latent = cond_refs_latent.unsqueeze(0) # 1 n_ref c h w
        # print('cond_refs_latent',cond_refs_latent.shape)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # if self.transformer.config.sample_size == 128:
        add_width = width*2
        resolution = torch.tensor([height, add_width]).repeat(batch_size * num_images_per_prompt, 1)
        aspect_ratio = torch.tensor([float(height / add_width)]).repeat(batch_size * num_images_per_prompt, 1)
        resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        # added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)


        K_cache = None
        V_cache = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                no_cache = K_cache is None and V_cache is None

                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                control_input = torch.concat([latent_model_input, cond_input_latent, hint_color_latent, hint_mask],1)

                control_list = self.controlnet(
                    control_input,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if no_cache:
                    noise_pred, K_cache, V_cache = self.transformer(
                        latent_model_input.to(dtype=self.transformer.dtype),
                        cond_refs_latent.to(dtype=self.transformer.dtype),
                        n_ref_lists = [num_ref_list],
                        encoder_hidden_states=prompt_embeds.to(dtype=self.transformer.dtype),
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        added_cond_kwargs=added_cond_kwargs,
                        control_list = control_list,
                        return_dict=False,
                        K_cache=None,
                        V_cache=None,
                    )
                else:
                    noise_pred, _, _ = self.transformer(
                        latent_model_input.to(dtype=self.transformer.dtype),
                        None,
                        n_ref_lists = None,
                        encoder_hidden_states=prompt_embeds.to(dtype=self.transformer.dtype),
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        added_cond_kwargs=added_cond_kwargs,
                        control_list = control_list,
                        return_dict=False,
                        K_cache=K_cache,
                        V_cache=V_cache,
                    )

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                if num_inference_steps == 1:

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample

                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        ref_out_idx = 0
        if not output_type == "latent":
            image = self.vae.decode(latents.to(dtype = self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)
        # Offload all models
        self.maybe_free_model_hooks()

        return (image,)
