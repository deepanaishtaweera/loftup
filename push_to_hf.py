"""
This script can be used to push the LoftUp upsamplers to the Hugging Face Hub.
"""

import torch

# push loftup-dinov2s
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2s', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-dinov2s")

# push loftup-siglip2
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_siglip2', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-siglip2")

# push loftup-siglip
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_siglip', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-siglip")

# push loftup-dinov2b
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2b', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-dinov2b")

# push loftup-dinov2s_reg
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2s_reg', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-dinov2s_reg")

# push loftup-dinov2b_reg
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2b_reg', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-dinov2b_reg")

# push loftup-clip
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_clip', pretrained=True)
upsampler.push_to_hub("haiwen/loftup-clip")
