# ComfyUI-NAG

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.

We currently support `Flux`, `SD3.5` and `SDXL`.

## Usage

To use NAG, simply replace the `CFGGuider` node in your workflow with `NAGCFGGuider`.

Example workflows are available in the `./workflows` directory!

![comfyui-nag](workflow.png)

## Key Inputs

We recommend tuning `nag_scale` for most use cases.

- `nag_scale`: The scale for attention feature extrapolation. Higher values result in stronger negative guidance.
- `nag_tau`: The normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
