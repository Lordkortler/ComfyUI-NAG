import torch
from comfy.ldm.modules.attention import CrossAttention, default, optimized_attention, optimized_attention_masked


class NAGCrossAttention(CrossAttention):
    def forward(
            self,
            x,
            context=None,
            value=None,
            mask=None,
            nag_scale: float = 1.0,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.5,
    ):
        origin_bsz = len(context) - len(x)
        assert origin_bsz != 0

        q = self.to_q(x)
        q = torch.cat([q, q[-origin_bsz:]], dim=0)

        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)

        # NAG
        out_negative, out_positive = out[-origin_bsz:], out[-origin_bsz * 2:-origin_bsz]
        out_guidance = out_positive * nag_scale - out_negative * (nag_scale - 1)
        norm_positive = torch.norm(out_positive, p=1, dim=-1, keepdim=True).expand(*out_positive.shape)
        norm_guidance = torch.norm(out_guidance, p=1, dim=-1, keepdim=True).expand(*out_guidance.shape)

        scale = norm_guidance / norm_positive
        out_guidance = out_guidance * torch.minimum(scale, scale.new_ones(1) * nag_tau) / scale

        out_guidance = out_guidance * nag_alpha + out_positive * (1 - nag_alpha)
        out = torch.cat([out[:-origin_bsz * 2], out_guidance], dim=0)

        return self.to_out(out)
