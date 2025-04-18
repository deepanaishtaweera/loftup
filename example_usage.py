from upsamplers import load_loftup_checkpoint, norm, unnorm
from featurizers import get_featurizer
from utils import plot_feats

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF

model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to('cuda')
kernel_size = 14
lr_size = 16
load_size = 224

upsampler_path = "/path/to/loftup_dinov2.ckpt"
upsampler = load_loftup_checkpoint(upsampler_path, 384, lr_pe_type="sine")
upsampler = upsampler.to('cuda')

image_path = "examples/sa_1.jpg"

transform = T.Compose([
        T.Resize(load_size, T.InterpolationMode.BILINEAR),
        T.CenterCrop(load_size), # Depending on whether you want a center crop
        T.ToTensor(),
        norm])
img = Image.open(image_path).convert("RGB")
normalized_img_tensor = transform(img).unsqueeze(0).to('cuda')
lr_feats = model.get_intermediate_layers(normalized_img_tensor, reshape=True)[0] # 1, 384, 14, 14

## Upsampling step
hr_feats = upsampler(lr_feats, normalized_img_tensor) # 1, 384, 224, 224

# ## You can also upsample to any shape you want; Just change the guidance image shape
# H, W = 112, 112
# img_tensor_112 = F.interpolate(normalized_img_tensor, size=(H, W), mode='bilinear', align_corners=False)
# hr_feats_112 = upsampler(lr_feats, img_tensor_112) # 1, 384, 112, 112

plot_feats(unnorm(normalized_img_tensor)[0], lr_feats[0], hr_feats[0], 'examples/feats.png')


#### Alternative usage with featurizer ####

model = get_featurizer("dinov2")
lr_feats = model(normalized_img_tensor)
hr_feats = upsampler(lr_feats, normalized_img_tensor)

## This should give the same result as the original usage