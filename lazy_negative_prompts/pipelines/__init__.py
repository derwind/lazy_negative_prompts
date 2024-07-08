from __future__ import annotations

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionControlNetPipeline
from .stable_diffusion import enable_stable_diffusion_lazy_negative
from .controlnet import enable_controlnet_lazy_negative


def enable_lazy_negative(pipe: DiffusionPipeline) -> bool:
    if isinstance(pipe, StableDiffusionPipeline):
        enable_stable_diffusion_lazy_negative()
        return True
    elif isinstance(pipe, StableDiffusionControlNetPipeline):
        enable_controlnet_lazy_negative()
        return True
    return False
