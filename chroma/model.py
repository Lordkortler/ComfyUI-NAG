from functools import partial
from types import MethodType

import torch
from torch import Tensor
from einops import rearrange, repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.chroma.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
)
from comfy.ldm.chroma.model import Chroma

from .layers import NAGDoubleStreamBlock, NAGSingleStreamBlock
from ..utils import cat_context


class NAGChroma(Chroma):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        origin_bsz = len(txt) - len(img)
        mod_vectors = torch.cat((mod_vectors, mod_vectors[-origin_bsz:]), dim=0)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
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
                                                               "vec": double_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=double_mod,
                                     pe=pe,
                                     attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = torch.cat((img, img[-origin_bsz:]), dim=0)
        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": single_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:-origin_bsz]
        img = img[:, txt.shape[1] :, ...]
        final_mod = self.get_modulations(mod_vectors[:-origin_bsz], "final")
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(
            self,
            x,
            timestep,
            context,
            guidance,
            control=None,
            transformer_options={},

            nag_negative_context=None,

            **kwargs,
    ):
        assert nag_negative_context is not None
        origin_context_len = context.shape[1]
        context = cat_context(context, nag_negative_context, trim_context=True)
        context_pad_len = context.shape[1] - origin_context_len
        nag_pad_len = context.shape[1] - nag_negative_context.shape[1]

        for block in self.double_blocks:
            block.forward = MethodType(
                partial(
                    NAGDoubleStreamBlock.forward,
                    context_pad_len=context_pad_len,
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
                    context_pad_len=context_pad_len,
                    nag_pad_len=nag_pad_len,
                ),
                block,
            )

        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]


def set_nag_chroma(model: Chroma, nag_negative_cond, nag_scale, nag_tau, nag_alpha):
    model.forward_orig = MethodType(NAGChroma.forward_orig, model)
    model.forward = MethodType(
        partial(
            NAGChroma.forward,
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


def set_origin_chroma(model: NAGChroma):
    model.forward_orig = MethodType(Chroma.forward_orig, model)
    model.forward = MethodType(Chroma.forward, model)
    for block in model.double_blocks:
        block.forward = MethodType(DoubleStreamBlock.forward, block)
    for block in model.single_blocks:
        block.forward = MethodType(SingleStreamBlock.forward, block)
