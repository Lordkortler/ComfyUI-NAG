from functools import partial
from types import MethodType
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import comfy.ldm.common_dit
from comfy.ldm.lumina.model import (
    NextDiT,
    JointTransformerBlock,
    JointAttention,
)
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.math import apply_rope

from ..utils import nag, cat_context, check_nag_activation, NAGSwitch


class NAGJointAttention(JointAttention):
    """NAG-enabled JointAttention for Lumina2/Z-Image models."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
        # NAG parameters
        origin_bsz: int = None,
        cap_len: int = None,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
    ) -> torch.Tensor:
        """
        Forward pass with NAG guidance.

        Args:
            x: Input tensor [batch_size + origin_bsz, seq_len, dim]
               - First batch_size samples are positive conditioning
               - Last origin_bsz samples are negative conditioning
            x_mask: Attention mask
            freqs_cis: RoPE frequencies
            transformer_options: Additional options
            origin_bsz: Number of samples in the negative batch
            cap_len: Length of caption tokens (to separate from image tokens)
            nag_scale: NAG scale parameter
            nag_tau: NAG tau parameter
            nag_alpha: NAG alpha parameter
        """
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Split positive and negative batches
        xq_positive = xq[:-origin_bsz]
        xk_positive = xk[:-origin_bsz]
        xv_positive = xv[:-origin_bsz]

        xq_negative = xq[-origin_bsz:]
        xk_negative = xk[-origin_bsz:]
        xv_negative = xv[-origin_bsz:]

        # Compute attention for positive batch
        output_positive = optimized_attention_masked(
            xq_positive.movedim(1, 2),
            xk_positive.movedim(1, 2),
            xv_positive.movedim(1, 2),
            self.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        # Compute attention for negative batch
        output_negative = optimized_attention_masked(
            xq_negative.movedim(1, 2),
            xk_negative.movedim(1, 2),
            xv_negative.movedim(1, 2),
            self.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        # Apply NAG only to image tokens (after cap_len)
        if cap_len is not None and cap_len > 0:
            # Split into caption and image parts
            output_positive_cap = output_positive[:, :cap_len]
            output_positive_img = output_positive[:, cap_len:]
            output_negative_img = output_negative[:, cap_len:]

            # Apply NAG to image tokens
            output_guidance_img = nag(
                output_positive_img[-origin_bsz:],
                output_negative_img,
                nag_scale,
                nag_tau,
                nag_alpha,
            )

            # Reconstruct output with guided image tokens
            output_positive_img = torch.cat(
                [output_positive_img[:-origin_bsz], output_guidance_img], dim=0
            )
            output = torch.cat([output_positive_cap, output_positive_img], dim=1)
        else:
            # Apply NAG to all tokens
            output_guidance = nag(
                output_positive[-origin_bsz:],
                output_negative,
                nag_scale,
                nag_tau,
                nag_alpha,
            )
            output = torch.cat([output_positive[:-origin_bsz], output_guidance], dim=0)

        return self.out(output)


class NAGJointTransformerBlock(JointTransformerBlock):
    """NAG-enabled JointTransformerBlock for Lumina2/Z-Image models."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        transformer_options={},
        # NAG parameters
        origin_bsz: int = None,
        cap_len: int = None,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
    ):
        """
        Forward pass with NAG guidance.
        """
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                adaln_input
            ).chunk(4, dim=1)

            # Get batch size for positive samples (excluding negative)
            positive_bsz = x.shape[0] - origin_bsz

            # Apply attention with NAG
            norm_x = self.attention_norm1(x)
            # Only modulate positive samples with their scale/shift
            modulated_x = norm_x.clone()
            modulated_x[:-origin_bsz] = norm_x[:-origin_bsz] * (
                1 + scale_msa[:-origin_bsz].unsqueeze(1)
            ) + 0  # modulate doesn't have shift
            modulated_x[-origin_bsz:] = norm_x[-origin_bsz:] * (
                1 + scale_msa[-origin_bsz:].unsqueeze(1)
            ) + 0

            attn_out = self.attention(
                modulated_x,
                x_mask,
                freqs_cis,
                transformer_options=transformer_options,
                origin_bsz=origin_bsz,
                cap_len=cap_len,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
            )

            # Only use positive batch gate and output
            x_out = x[:-origin_bsz] + gate_msa[:-origin_bsz].unsqueeze(1).tanh() * self.attention_norm2(attn_out)

            # FFN (only on positive batch)
            norm_x_ffn = self.ffn_norm1(x_out)
            modulated_x_ffn = norm_x_ffn * (1 + scale_mlp[:-origin_bsz].unsqueeze(1))
            x_out = x_out + gate_mlp[:-origin_bsz].unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(modulated_x_ffn)
            )
        else:
            assert adaln_input is None
            attn_out = self.attention(
                self.attention_norm1(x),
                x_mask,
                freqs_cis,
                transformer_options=transformer_options,
                origin_bsz=origin_bsz,
                cap_len=cap_len,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
            )
            x_out = x[:-origin_bsz] + self.attention_norm2(attn_out)
            x_out = x_out + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x_out)))

        return x_out


class NAGNextDiT(NextDiT):
    """NAG-enabled NextDiT model for Lumina2/Z-Image."""

    def forward_nag(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        nag_negative_context=None,
        nag_sigma_end=0.0,
        nag_scale=1.0,
        nag_tau=2.5,
        nag_alpha=0.25,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        # Time embedding
        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)

        # Concatenate positive and negative contexts
        origin_bsz = nag_negative_context.shape[0]
        cap_feats_combined = cat_context(cap_feats, nag_negative_context, trim_context=True)

        # Embed captions
        cap_feats_embedded = self.cap_embedder(cap_feats_combined)

        # Process padding for Z-Image models with pad_tokens_multiple
        bsz = bs
        pH = pW = self.patch_size
        device = x.device

        if self.pad_tokens_multiple is not None:
            pad_extra = (-cap_feats_embedded.shape[1]) % self.pad_tokens_multiple
            cap_feats_embedded = torch.cat(
                (
                    cap_feats_embedded,
                    self.cap_pad_token.to(
                        device=cap_feats_embedded.device,
                        dtype=cap_feats_embedded.dtype,
                        copy=True,
                    )
                    .unsqueeze(0)
                    .repeat(cap_feats_embedded.shape[0], pad_extra, 1),
                ),
                dim=1,
            )

        cap_pos_ids = torch.zeros(
            cap_feats_embedded.shape[0], cap_feats_embedded.shape[1], 3, dtype=torch.float32, device=device
        )
        cap_pos_ids[:, :, 0] = (
            torch.arange(cap_feats_embedded.shape[1], dtype=torch.float32, device=device) + 1.0
        )

        # Process image
        B, C, H, W = x.shape
        x_embedded = self.x_embedder(
            x.view(B, C, H // pH, pH, W // pW, pW)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )

        H_tokens, W_tokens = H // pH, W // pW
        x_pos_ids = torch.zeros(
            (bsz, x_embedded.shape[1], 3), dtype=torch.float32, device=device
        )
        x_pos_ids[:, :, 0] = cap_feats_embedded.shape[1] + 1
        x_pos_ids[:, :, 1] = (
            torch.arange(H_tokens, dtype=torch.float32, device=device)
            .view(-1, 1)
            .repeat(1, W_tokens)
            .flatten()
        )
        x_pos_ids[:, :, 2] = (
            torch.arange(W_tokens, dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(H_tokens, 1)
            .flatten()
        )

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x_embedded.shape[1]) % self.pad_tokens_multiple
            x_embedded = torch.cat(
                (
                    x_embedded,
                    self.x_pad_token.to(
                        device=x_embedded.device, dtype=x_embedded.dtype, copy=True
                    )
                    .unsqueeze(0)
                    .repeat(x_embedded.shape[0], pad_extra, 1),
                ),
                dim=1,
            )
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        # Expand time embedding for negative batch
        t_emb_expanded = torch.cat([t_emb, t_emb[-origin_bsz:]], dim=0)
        adaln_input = t_emb_expanded

        # Build RoPE for positive samples
        cap_pos_ids_positive = cap_pos_ids[:bsz]
        freqs_cis_positive = self.rope_embedder(
            torch.cat((cap_pos_ids_positive, x_pos_ids), dim=1)
        ).movedim(1, 2)

        # Build RoPE for negative samples
        cap_pos_ids_negative = cap_pos_ids[bsz : bsz + origin_bsz]
        x_pos_ids_negative = x_pos_ids[-origin_bsz:] if x_pos_ids.shape[0] > bsz else x_pos_ids[:origin_bsz]
        freqs_cis_negative = self.rope_embedder(
            torch.cat((cap_pos_ids_negative, x_pos_ids_negative), dim=1)
        ).movedim(1, 2)

        # Context refiner (operates on caption only) - process positive and negative separately
        cap_feats_positive = cap_feats_embedded[:bsz]
        cap_feats_negative = cap_feats_embedded[bsz : bsz + origin_bsz]

        for layer in self.context_refiner:
            cap_feats_positive = layer(
                cap_feats_positive,
                cap_mask,
                freqs_cis_positive[:, : cap_pos_ids_positive.shape[1]],
                transformer_options=transformer_options,
            )
            cap_feats_negative = layer(
                cap_feats_negative,
                cap_mask,
                freqs_cis_negative[:, : cap_pos_ids_negative.shape[1]],
                transformer_options=transformer_options,
            )

        cap_len = cap_feats_positive.shape[1]

        # Noise refiner (operates on image only) - only on positive
        padded_img_mask = None
        for layer in self.noise_refiner:
            x_embedded = layer(
                x_embedded,
                padded_img_mask,
                freqs_cis_positive[:, cap_pos_ids_positive.shape[1] :],
                t_emb,
                transformer_options=transformer_options,
            )

        # Concatenate caption and image for main layers
        # Positive: [cap_feats_positive, x_embedded]
        # Negative: [cap_feats_negative, x_embedded_copy]
        padded_full_embed_positive = torch.cat((cap_feats_positive, x_embedded), dim=1)
        x_embedded_for_negative = x_embedded[-origin_bsz:] if x_embedded.shape[0] >= origin_bsz else x_embedded[:origin_bsz].clone()
        padded_full_embed_negative = torch.cat(
            (cap_feats_negative, x_embedded_for_negative), dim=1
        )

        # Combined for NAG processing
        padded_full_embed = torch.cat(
            [padded_full_embed_positive, padded_full_embed_negative], dim=0
        )

        # Concatenate freqs_cis
        freqs_cis_combined = torch.cat([freqs_cis_positive, freqs_cis_negative], dim=0)

        mask = None
        img_sizes = [(H, W)] * bsz

        # Main transformer layers with NAG
        for layer in self.layers:
            padded_full_embed = self._forward_layer_with_nag(
                layer,
                padded_full_embed,
                mask,
                freqs_cis_combined,
                adaln_input,
                transformer_options,
                origin_bsz=origin_bsz,
                cap_len=cap_len,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
            )
            # After NAG, we only have positive batch
            # Need to reconstruct for next layer
            padded_full_embed = torch.cat(
                [padded_full_embed, padded_full_embed[-origin_bsz:]], dim=0
            )

        # Remove negative batch for final processing
        padded_full_embed = padded_full_embed[:-origin_bsz]

        # Final layer
        x_out = self.final_layer(padded_full_embed, t_emb)

        # Unpatchify
        l_effective_cap_len = [cap_len] * bsz
        x_out = self.unpatchify(x_out, img_sizes, l_effective_cap_len, return_tensor=True)[
            :, :, :h, :w
        ]

        return -x_out

    def _forward_layer_with_nag(
        self,
        layer: JointTransformerBlock,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor,
        transformer_options: dict,
        origin_bsz: int,
        cap_len: int,
        nag_scale: float,
        nag_tau: float,
        nag_alpha: float,
    ):
        """Forward through a single layer with NAG guidance."""
        if layer.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = layer.adaLN_modulation(
                adaln_input
            ).chunk(4, dim=1)

            # Normalize and modulate
            norm_x = layer.attention_norm1(x)
            modulated_x = norm_x * (1 + scale_msa.unsqueeze(1))

            # Compute attention with NAG
            attn_out = self._forward_attention_with_nag(
                layer.attention,
                modulated_x,
                x_mask,
                freqs_cis,
                transformer_options,
                origin_bsz=origin_bsz,
                cap_len=cap_len,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
            )

            # Apply gate and residual (only positive batch)
            x_out = x[:-origin_bsz] + gate_msa[:-origin_bsz].unsqueeze(1).tanh() * layer.attention_norm2(attn_out)

            # FFN
            norm_x_ffn = layer.ffn_norm1(x_out)
            modulated_x_ffn = norm_x_ffn * (1 + scale_mlp[:-origin_bsz].unsqueeze(1))
            x_out = x_out + gate_mlp[:-origin_bsz].unsqueeze(1).tanh() * layer.ffn_norm2(
                layer.feed_forward(modulated_x_ffn)
            )
        else:
            attn_out = self._forward_attention_with_nag(
                layer.attention,
                layer.attention_norm1(x),
                x_mask,
                freqs_cis,
                transformer_options,
                origin_bsz=origin_bsz,
                cap_len=cap_len,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
            )
            x_out = x[:-origin_bsz] + layer.attention_norm2(attn_out)
            x_out = x_out + layer.ffn_norm2(
                layer.feed_forward(layer.ffn_norm1(x_out))
            )

        return x_out

    def _forward_attention_with_nag(
        self,
        attention: JointAttention,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options: dict,
        origin_bsz: int,
        cap_len: int,
        nag_scale: float,
        nag_tau: float,
        nag_alpha: float,
    ):
        """Forward through attention with NAG guidance."""
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            attention.qkv(x),
            [
                attention.n_local_heads * attention.head_dim,
                attention.n_local_kv_heads * attention.head_dim,
                attention.n_local_kv_heads * attention.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, attention.n_local_heads, attention.head_dim)
        xk = xk.view(bsz, seqlen, attention.n_local_kv_heads, attention.head_dim)
        xv = xv.view(bsz, seqlen, attention.n_local_kv_heads, attention.head_dim)

        xq = attention.q_norm(xq)
        xk = attention.k_norm(xk)

        # Apply RoPE separately for positive and negative
        xq_positive, xk_positive = apply_rope(
            xq[:-origin_bsz], xk[:-origin_bsz], freqs_cis[:-origin_bsz]
        )
        xq_negative, xk_negative = apply_rope(
            xq[-origin_bsz:], xk[-origin_bsz:], freqs_cis[-origin_bsz:]
        )

        xv_positive = xv[:-origin_bsz]
        xv_negative = xv[-origin_bsz:]

        n_rep = attention.n_local_heads // attention.n_local_kv_heads
        if n_rep >= 1:
            xk_positive = xk_positive.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv_positive = xv_positive.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xk_negative = xk_negative.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv_negative = xv_negative.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Compute attention for positive batch
        output_positive = optimized_attention_masked(
            xq_positive.movedim(1, 2),
            xk_positive.movedim(1, 2),
            xv_positive.movedim(1, 2),
            attention.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        # Compute attention for negative batch
        output_negative = optimized_attention_masked(
            xq_negative.movedim(1, 2),
            xk_negative.movedim(1, 2),
            xv_negative.movedim(1, 2),
            attention.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        # Apply NAG only to image tokens (after cap_len)
        if cap_len is not None and cap_len > 0:
            # Split into caption and image parts
            output_positive_cap = output_positive[:, :cap_len]
            output_positive_img = output_positive[:, cap_len:]
            output_negative_img = output_negative[:, cap_len:]

            # Apply NAG to image tokens
            output_guidance_img = nag(
                output_positive_img[-origin_bsz:],
                output_negative_img,
                nag_scale,
                nag_tau,
                nag_alpha,
            )

            # Reconstruct output with guided image tokens
            output_positive_img = torch.cat(
                [output_positive_img[:-origin_bsz], output_guidance_img], dim=0
            )
            output = torch.cat([output_positive_cap, output_positive_img], dim=1)
        else:
            # Apply NAG to all tokens
            output_guidance = nag(
                output_positive[-origin_bsz:],
                output_negative,
                nag_scale,
                nag_tau,
                nag_alpha,
            )
            output = torch.cat([output_positive[:-origin_bsz], output_guidance], dim=0)

        return attention.out(output)

    def forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        nag_negative_context=None,
        nag_sigma_end=0.0,
        nag_scale=1.0,
        nag_tau=2.5,
        nag_alpha=0.25,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})

        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)

        if apply_nag and nag_negative_context is not None:
            return self.forward_nag(
                x,
                timesteps,
                context,
                num_tokens,
                attention_mask,
                nag_negative_context=nag_negative_context,
                nag_sigma_end=nag_sigma_end,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
                **kwargs,
            )
        else:
            # Fall back to original forward
            return self.forward_orig(
                x, timesteps, context, num_tokens, attention_mask, **kwargs
            )


class NAGNextDiTSwitch(NAGSwitch):
    """Switch class to enable/disable NAG for Lumina2/Z-Image models."""

    def set_nag(self):
        # Store original forward
        self.model.forward_orig = self.model._forward

        # Replace forward with NAG-enabled version
        self.model.forward = MethodType(
            partial(
                NAGNextDiT.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
                nag_scale=self.nag_scale,
                nag_tau=self.nag_tau,
                nag_alpha=self.nag_alpha,
            ),
            self.model,
        )

        # Also bind the helper methods
        self.model.forward_nag = MethodType(NAGNextDiT.forward_nag, self.model)
        self.model._forward_layer_with_nag = MethodType(
            NAGNextDiT._forward_layer_with_nag, self.model
        )
        self.model._forward_attention_with_nag = MethodType(
            NAGNextDiT._forward_attention_with_nag, self.model
        )

    def set_origin(self):
        super().set_origin()
        if hasattr(self.model, "forward_orig"):
            del self.model.forward_orig
        if hasattr(self.model, "forward_nag"):
            del self.model.forward_nag
        if hasattr(self.model, "_forward_layer_with_nag"):
            del self.model._forward_layer_with_nag
        if hasattr(self.model, "_forward_attention_with_nag"):
            del self.model._forward_attention_with_nag
