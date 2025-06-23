from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import torch
from comfy.samplers import (
    process_conds,
    preprocess_conds_hooks,
    cast_to_load_options,
    filter_registered_hooks_on_conds,
    get_total_hook_groups_in_conds,
    CFGGuider,
)
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
from comfy.ldm.flux.model import Flux
from comfy.ldm.chroma.model import Chroma
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper

from .flux.model import set_nag_flux, set_origin_flux
from .chroma.model import set_nag_chroma, set_origin_chroma
from .sd.openaimodel import set_nag_sd, set_origin_sd
from .sd3.mmdit import set_nag_sd3, set_origin_sd3


class NAGCFGGuider(CFGGuider):
    def __init__(self, model_patcher: ModelPatcher):
        super().__init__(model_patcher=model_patcher)
        self.origin_nag_negative_cond = None
        self.nag_scale = 5.0
        self.nag_tau = 3.5
        self.nag_alpha = 0.25

    def set_nag(self, nag_negative_cond, nag_scale, nag_tau, nag_alpha):
        self.origin_nag_negative_cond = nag_negative_cond
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        apply_guidance = self.nag_scale > 1.

        self.nag_negative_cond = None
        if apply_guidance:
            self.nag_negative_cond = copy.deepcopy(self.origin_nag_negative_cond)

            model = self.model_patcher.model.diffusion_model
            model_type = type(model)
            if model_type == Flux:
                set_fn = set_nag_flux
                reset_fn = set_origin_flux
            elif model_type == Chroma:
                set_fn = set_nag_chroma
                reset_fn = set_origin_chroma
            elif model_type == UNetModel:
                set_fn = set_nag_sd
                reset_fn = set_origin_sd
            elif model_type == OpenAISignatureMMDITWrapper:
                set_fn = set_nag_sd3
                reset_fn = set_origin_sd3
            else:
                raise ValueError(
                    f"Model type {model_type} is not support for NAGCFGGuider"
                )
            set_fn(model, self.nag_negative_cond, self.nag_scale, self.nag_tau, self.nag_alpha)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        if apply_guidance:
            reset_fn(model)

        del self.conds
        del self.nag_negative_cond
        return output
