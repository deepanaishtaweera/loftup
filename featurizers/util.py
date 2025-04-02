import torch

def get_featurizer(name, activation_type="token", **kwargs):
    name = name.lower()
    if name == "dinov2":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer("dinov2_vits14", patch_size, activation_type)
        dim = 384
    elif name == "maskclip":
        from .MaskCLIP import MaskCLIPFeaturizer
        patch_size = 16
        model = MaskCLIPFeaturizer()
        dim = 512
    elif name == "sigclip":
        from .SigCLIP import SigLIPFeaturizer
        patch_size = 16
        model = SigLIPFeaturizer("hf-hub:timm/ViT-B-16-SigLIP", patch_size)
        dim = 768
    elif name == "sigclip2":
        from .SigCLIP import SigLIPFeaturizer
        patch_size = 16
        model = SigLIPFeaturizer("hf-hub:timm/ViT-B-16-SigLIP2", patch_size)
        dim = 768
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim
