import math
import torch


def nag(z_positive, z_negative, scale, tau, alpha):
    z_guidance = z_positive * scale - z_negative * (scale - 1)
    norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True).expand(*z_positive.shape)
    norm_guidance = torch.norm(z_guidance, p=1, dim=-1, keepdim=True).expand(*z_guidance.shape)

    scale = norm_guidance / norm_positive
    z_guidance = z_guidance * torch.minimum(scale, scale.new_ones(1) * tau) / scale

    z_guidance = z_guidance * alpha + z_positive * (1 - alpha)
    
    return z_guidance


def cat_context(context, nag_negative_context, trim_context=False):
    nag_negative_context = nag_negative_context.to(context)

    context_len = context.shape[1]
    nag_neg_context_len = nag_negative_context.shape[1]

    if context_len < nag_neg_context_len:
        context = context.repeat(1, math.ceil(nag_neg_context_len / context_len), 1)
        if trim_context:
            context = context[:, -nag_neg_context_len:]
        context_len = context.shape[1]

    nag_negative_context = nag_negative_context.repeat(1, math.ceil(context_len / nag_neg_context_len), 1)
    nag_negative_context = nag_negative_context[:, -context_len:]

    return torch.cat([context, nag_negative_context], dim=0)


def check_nag_activation(transformer_options, nag_sigma_end):
    apply_nag = torch.all(transformer_options["sigmas"] >= nag_sigma_end)
    positive_batch = 0 in transformer_options["cond_or_uncond"]
    return apply_nag and positive_batch


def get_closure_vars(func):
    if func.__closure__ is None:
        return {}
    return {
        var: cell.cell_contents
        for var, cell in zip(func.__code__.co_freevars, func.__closure__)
    }


def is_from_wavespeed(func):
    closure = get_closure_vars(func)
    return "residual_diff_threshold" in closure \
        and "validate_can_use_cache_function" in closure


# https://github.com/welltop-cn/ComfyUI-TeaCache/blob/4bca908bf53b029ea5739cb69ef2a9e6c06e6752/nodes.py
def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result
