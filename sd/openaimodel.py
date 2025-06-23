from functools import partial
from types import MethodType

import torch
import comfy
from comfy.ldm.modules.attention import CrossAttention
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from .attention import NAGCrossAttention


class NAGUnetModel(UNetModel):
    def forward(
            self,
            x,
            timesteps=None,
            context=None,
            y=None,
            control=None,
            transformer_options={},

            nag_negative_context=None,

            **kwargs,
    ):
        assert nag_negative_context is not None
        context = torch.cat((context, nag_negative_context[:, :context.shape[1]].to(context)), dim=0)

        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                                     transformer_options)
        ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)


def set_nag_sd(model: UNetModel, nag_negative_cond, nag_scale, nag_tau, nag_alpha):
    model.forward = MethodType(
        partial(
            NAGUnetModel.forward,
            nag_negative_context=nag_negative_cond[0][0],
        ),
        model
    )
    for name, module in model.named_modules():
        if "attn2" in name and isinstance(module, CrossAttention):
            module.forward = MethodType(
                partial(
                    NAGCrossAttention.forward,
                    nag_scale=nag_scale,
                    nag_tau=nag_tau,
                    nag_alpha=nag_alpha,
                ),
                module,
            )


def set_origin_sd(model: NAGUnetModel):
    model.forward = MethodType(UNetModel.forward, model)
    for name, module in model.named_modules():
        if "attn2" in name and isinstance(module, CrossAttention):
            module.forward = MethodType(CrossAttention.forward, module)
