import comfy
from .samplers import KSamplerWithNAG


def sample_with_nag(
        model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, sampler_name, scheduler, positive, negative, nag_negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    sampler = KSamplerWithNAG(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(
        noise, positive, negative, nag_negative,
        cfg=cfg, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha,
        latent_image=latent_image,
        start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
        denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples
