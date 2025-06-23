from functools import partial
from types import MethodType

import torch
from torch import Tensor
from einops import rearrange, repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    timestep_embedding,
)
from comfy.ldm.flux.model import Flux

from .layers import NAGDoubleStreamBlock, NAGSingleStreamBlock
from ..utils import cat_context


class NAGFlux(Flux):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((img, img[-origin_bsz:]), dim=0)
        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:-origin_bsz]
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec[:-origin_bsz])  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(
            self,
            x,
            timestep,
            context,
            y=None,
            guidance=None,
            control=None,
            transformer_options={},

            nag_negative_context=None,
            nag_negative_y=None,

            **kwargs):
        assert nag_negative_context is not None and nag_negative_y is not None
        context = cat_context(context, nag_negative_context)
        y = torch.cat((y, nag_negative_y.to(y)), dim=0)
        nag_pad_len = context.shape[1] - nag_negative_context.shape[1]

        for block in self.double_blocks:
            block.forward = MethodType(
                partial(
                    NAGDoubleStreamBlock.forward,
                    nag_pad_len=nag_pad_len,
                ),
                block,
            )
        for block in self.single_blocks:
            block.forward = MethodType(
                partial(
                    NAGSingleStreamBlock.forward,
                    txt_length=context.shape[1],
                    origin_bsz=nag_negative_context.shape[0],
                    nag_pad_len=nag_pad_len,
                ),
                block,
            )

        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]


def set_nag_flux(model: Flux, nag_negative_cond, nag_scale, nag_tau, nag_alpha):
    model.forward_orig = MethodType(NAGFlux.forward_orig, model)
    model.forward = MethodType(
        partial(
            NAGFlux.forward,
            nag_negative_context=nag_negative_cond[0][0],
            nag_negative_y=nag_negative_cond[0][1]["pooled_output"],
        ),
        model
    )
    for block in model.double_blocks:
        block.nag_scale = nag_scale
        block.nag_tau = nag_tau
        block.nag_alpha = nag_alpha
        block.forward = MethodType(NAGDoubleStreamBlock.forward, block)
    for block in model.single_blocks:
        block.nag_scale = nag_scale
        block.nag_tau = nag_tau
        block.nag_alpha = nag_alpha
        block.forward = MethodType(NAGSingleStreamBlock.forward, block)


def set_origin_flux(model: NAGFlux):
    model.forward_orig = MethodType(Flux.forward_orig, model)
    model.forward = MethodType(Flux.forward, model)
    for block in model.double_blocks:
        block.forward = MethodType(DoubleStreamBlock.forward, block)
    for block in model.single_blocks:
        block.forward = MethodType(SingleStreamBlock.forward, block)
