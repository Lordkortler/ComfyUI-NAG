from types import MethodType
from functools import partial

import torch
from torch import Tensor

from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock, timestep_embedding
from comfy.ldm.hunyuan_video.model import HunyuanVideo

from ..flux.layers import NAGSingleStreamBlock, NAGDoubleStreamBlock
from ..utils import cat_context


class NAGHunyuanVideo(HunyuanVideo):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                guidance = torch.cat((guidance, guidance[-origin_bsz:]), dim=0)
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        pe = torch.cat((pe, pe[-origin_bsz:]), dim=0)
        img = torch.cat((img, img[-origin_bsz:]), dim=0)
        img = torch.cat((img, txt), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        img = img[:-origin_bsz]
        img = img[:, : img_len]
        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]

        img = self.final_layer(img, vec[:-origin_bsz], modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

    def forward(
            self,
            x,
            timestep,
            context,
            y,
            guidance=None,
            attention_mask=None,
            guiding_frame_index=None,
            ref_latent=None,
            control=None,
            transformer_options={},

            positive_context=None,
            nag_negative_context=None,
            nag_negative_y=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        bs, c, t, h, w = x.shape
        img_ids = self.img_ids(x)

        apply_nag = transformer_options["sigmas"] >= nag_sigma_end
        positive_batch = \
            context.shape[0] != nag_negative_context.shape[0] \
            or context.shape[1] == positive_context.shape[1] and torch.all(torch.isclose(context, positive_context.to(context)))
        if apply_nag and positive_batch:
            origin_context_len = context.shape[1]
            context = cat_context(context, nag_negative_context, trim_context=True)
            y = torch.cat((y, nag_negative_y.to(y)), dim=0)
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context.shape[1]

            self.forward_orig = MethodType(NAGHunyuanVideo.forward_orig, self)
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
                        img_length=img_ids.shape[1],
                        origin_bsz=nag_negative_context.shape[0],
                        context_pad_len=context_pad_len,
                        nag_pad_len=nag_pad_len,
                    ),
                    block,
                )
        else:
            self.forward_orig = MethodType(HunyuanVideo.forward_orig, self)
            for block in self.double_blocks:
                block.forward = MethodType(DoubleStreamBlock.forward, block)
            for block in self.single_blocks:
                block.forward = MethodType(SingleStreamBlock.forward, block)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(x, img_ids, context, txt_ids, attention_mask, timestep, y, guidance,
                                guiding_frame_index, ref_latent, control=control,
                                transformer_options=transformer_options)
        return out


def set_nag_hunyuan_video(
        model: HunyuanVideo,
        positive_context,
        nag_negative_cond,
        nag_scale, nag_tau, nag_alpha, nag_sigma_end,
):
    model.forward = MethodType(
        partial(
            NAGHunyuanVideo.forward,
            positive_context=positive_context,
            nag_negative_context=nag_negative_cond[0][0],
            nag_negative_y=nag_negative_cond[0][1]["pooled_output"],
            nag_sigma_end=nag_sigma_end,
        ),
        model,
    )
    for block in model.double_blocks:
        block.nag_scale = nag_scale
        block.nag_tau = nag_tau
        block.nag_alpha = nag_alpha
    for block in model.single_blocks:
        block.nag_scale = nag_scale
        block.nag_tau = nag_tau
        block.nag_alpha = nag_alpha


def set_origin_hunyuan_video(model: NAGHunyuanVideo):
    model.forward_orig = MethodType(HunyuanVideo.forward_orig, model)
    model.forward = MethodType(HunyuanVideo.forward, model)
    for block in model.double_blocks:
        block.forward = MethodType(DoubleStreamBlock.forward, block)
    for block in model.single_blocks:
        block.forward = MethodType(SingleStreamBlock.forward, block)

