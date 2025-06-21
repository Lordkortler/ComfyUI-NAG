import torch
from torch import Tensor

from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, apply_mod


class NAGDoubleStreamBlock(DoubleStreamBlock):
    def forward(
            self,
            img: Tensor,
            txt: Tensor,
            vec: Tensor,
            pe: Tensor,
            attn_mask=None,
            modulation_dims_img=None,
            modulation_dims_txt=None,
            nag_scale: float = 1.0,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
    ):
        origin_bsz = len(txt) - len(img)
        assert origin_bsz != 0

        img_mod1, img_mod2 = self.img_mod(vec[:-origin_bsz])
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        img_q = torch.cat((img_q, img_q[-origin_bsz:]), dim=0)
        img_k = torch.cat((img_k, img_k[-origin_bsz:]), dim=0)
        img_v = torch.cat((img_v, img_v[-origin_bsz:]), dim=0)

        if self.flipped_img_txt:
            # run actual attention
            attn = attention(torch.cat((img_q, txt_q), dim=2),
                             torch.cat((img_k, txt_k), dim=2),
                             torch.cat((img_v, txt_v), dim=2),
                             pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
        else:
            # run actual attention
            attn = attention(torch.cat((txt_q, img_q), dim=2),
                             torch.cat((txt_k, img_k), dim=2),
                             torch.cat((txt_v, img_v), dim=2),
                             pe=pe, mask=attn_mask)

            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # NAG
        img_attn_negative, img_attn_positive = img_attn[-origin_bsz:], img_attn[-origin_bsz * 2:-origin_bsz]

        img_attn_guidance = img_attn_positive * nag_scale - img_attn_negative * (nag_scale - 1)
        norm_positive = torch.norm(img_attn_positive, p=2, dim=-1, keepdim=True).expand(*img_attn_positive.shape)
        norm_guidance = torch.norm(img_attn_guidance, p=2, dim=-1, keepdim=True).expand(*img_attn_positive.shape)

        scale = norm_guidance / norm_positive
        img_attn_guidance = img_attn_guidance * torch.minimum(scale, scale.new_ones(1) * nag_tau) / scale

        img_attn_guidance = img_attn_guidance * nag_alpha + img_attn_positive * (1 - nag_alpha)

        img_attn = torch.cat([img_attn[:-origin_bsz * 2], img_attn_guidance], dim=0)

        # calculate the img bloks
        img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        img = img + apply_mod(
            self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)),
            img_mod2.gate, None, modulation_dims_img)

        # calculate the txt bloks
        txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        txt += apply_mod(
            self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)),
            txt_mod2.gate, None, modulation_dims_txt)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class NAGSingleStreamBlock(SingleStreamBlock):
    def forward(
            self,
            x: Tensor,
            vec: Tensor,
            pe: Tensor,
            attn_mask=None,
            modulation_dims=None,

            nag_scale: float = 1.0,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
            txt_length:int = None,
            origin_bsz: int = None,
    ) -> Tensor:
        mod= self.modulation(vec)[0]
        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # NAG
        q, q_negative = q[:-origin_bsz], q[-origin_bsz:]
        k, k_negative = k[:-origin_bsz], k[-origin_bsz:]
        v, v_negative = v[:-origin_bsz], v[-origin_bsz:]

        # compute attention
        attn_negative = attention(q_negative, k_negative, v_negative, pe=pe, mask=attn_mask)
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        img_attn_negative = attn_negative[:, txt_length:]
        img_attn = attn[:, txt_length:]

        img_attn_positive = img_attn[-origin_bsz:]

        img_attn_guidance = img_attn_positive * nag_scale - img_attn_negative * (nag_scale - 1)
        norm_positive = torch.norm(img_attn_positive, p=2, dim=-1, keepdim=True).expand(*img_attn_positive.shape)
        norm_guidance = torch.norm(img_attn_guidance, p=2, dim=-1, keepdim=True).expand(*img_attn_positive.shape)

        scale = norm_guidance / norm_positive
        img_attn_guidance = img_attn_guidance * torch.minimum(scale, scale.new_ones(1) * nag_tau) / scale

        img_attn_guidance = img_attn_guidance * nag_alpha + img_attn_positive * (1 - nag_alpha)

        attn_negative[:, txt_length:] = img_attn_guidance
        attn[-origin_bsz:, txt_length:] = img_attn_guidance

        # compute activation in mlp stream, cat again and run second linear layer
        output_negative = self.linear2(torch.cat((attn_negative, self.mlp_act(mlp[-origin_bsz:])), 2))
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp[:-origin_bsz])), 2))

        output = torch.cat([output, output_negative], dim=0)

        x += apply_mod(output, mod.gate, None, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x
