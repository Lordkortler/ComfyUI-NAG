import math
import torch


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
