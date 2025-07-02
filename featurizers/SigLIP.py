import math
import warnings
from functools import partial

import timm
import torch
import torch.nn as nn

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from open_clip import create_model_from_pretrained, tokenize # works on open-clip-torch>=2.23.0, timm>=0.9.8


class SigLIPFeaturizer(nn.Module):

    def __init__(self, arch='hf-hub:timm/ViT-B-16-SigLIP', patch_size=16, feat_type=None):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        self.dim = 768
        siglip, _ = create_model_from_pretrained(arch)
        self.model = siglip.visual.trunk
        self.model_text = siglip.text

    def forward(self, img, n=1, include_cls=False):
        features = self.model.forward_features(img) # b, hxw, c
        h, w = img.shape[-2] // self.patch_size, img.shape[-1] // self.patch_size
        features = features.permute(0, 2, 1).reshape(-1, self.dim, h, w)
        return features

    def forward_text(self, texts):
        """
        Encode a list of texts and return their embeddings.
        Args:
            texts (list of str): The input texts.
        Returns:
            torch.Tensor: The text embeddings.
        """
        device = next(self.model_text.parameters()).device
        context_length = getattr(self.model_text, 'context_length', 77)  # fallback to 77 if not present
        tokens = tokenize(texts, context_length=context_length).to(device)
        with torch.no_grad():
            embeddings = self.model_text(tokens)
        return embeddings