from functools import partial
from types import MethodType

import torch
import comfy
from comfy.ldm.modules.attention import CrossAttention
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from .attention import NAGCrossAttention
from ..utils import cat_context


class NAGUnetModel(UNetModel):
    def forward(
            self,
            x,
            timesteps=None,
            context=None,
            y=None,
            control=None,
            transformer_options={},

            positive_context=None,
            nag_negative_context=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        apply_nag = transformer_options["sigmas"] >= nag_sigma_end
        positive_batch = \
            context.shape[0] != nag_negative_context.shape[0] \
            or context.shape[1] == positive_context.shape[1] and torch.all(torch.isclose(context, positive_context.to(context)))
        if apply_nag and positive_batch:
            context = cat_context(context, nag_negative_context)
            for name, module in self.named_modules():
                if "attn2" in name and isinstance(module, CrossAttention):
                    module.forward = MethodType(NAGCrossAttention.forward, module)
        else:
            for name, module in self.named_modules():
                if "attn2" in name and isinstance(module, CrossAttention):
                    module.forward = MethodType(CrossAttention.forward, module)

        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                                     transformer_options)
        ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)


def set_nag_sd(
        model: UNetModel,
        positive_context,
        nag_negative_cond,
        nag_scale, nag_tau, nag_alpha, nag_sigma_end,
):
    model.forward = MethodType(
        partial(
            NAGUnetModel.forward,
            positive_context=positive_context,
            nag_negative_context=nag_negative_cond[0][0],
            nag_sigma_end=nag_sigma_end,
        ),
        model
    )
    for name, module in model.named_modules():
        if "attn2" in name and isinstance(module, CrossAttention):
            module.nag_scale = nag_scale
            module.nag_tau = nag_tau
            module.nag_alpha = nag_alpha


def set_origin_sd(model: NAGUnetModel):
    model.forward = MethodType(UNetModel.forward, model)
    for name, module in model.named_modules():
        if "attn2" in name and isinstance(module, CrossAttention):
            module.forward = MethodType(CrossAttention.forward, module)
