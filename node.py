from .samplers import NAGCFGGuider as samplers_NAGCFGGuider


class NAGCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "nag_negative": ("CONDITIONING", ),
                        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step":0.1, "round": 0.01}),
                        "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                        "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            nag_negative,
            cfg,
            nag_scale,
            nag_tau,
            nag_alpha,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "NAGCFGGuider": NAGCFGGuider,
}
