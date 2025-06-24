# ComfyUI-NAG

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.

Paper: https://arxiv.org/abs/2505.21179

Code: https://github.com/ChenDarYen/Normalized-Attention-Guidance

Wan2.1 Demo: https://huggingface.co/spaces/ChenDY/NAG_wan2-1-fast

LTX Video Demo: https://huggingface.co/spaces/ChenDY/NAG_ltx-video-distilled

Flux-Dev Demo: https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev

## Nodes

- `NAGCFGGuider`
- `KSamplerWithNAG`

## Usage

To use NAG, simply replace the `CFGGuider` node with `NAGCFGGuider`, or the `KSampler` node with `KSamplerWithNAG` in your workflow.

We currently support `Flux`, `Choroma`, `SD3.5` and `SDXL`.

Example workflows are available in the `./workflows` directory!

![comfyui-nag](workflow.png)

## Key Inputs

We recommend tuning `nag_scale` for most use cases.

- `nag_scale`: The scale for attention feature extrapolation. Higher values result in stronger negative guidance.
- `nag_tau`: The normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
