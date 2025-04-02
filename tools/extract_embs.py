import hydra
# import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from os.path import join
import numpy as np

from featup.datasets.util import get_dataset, SingleImageDataset
from featup.featurizers.util import get_featurizer

from featup.util import pca, RollingAvg, unnorm, norm, prep_image
from tqdm import tqdm
import os
import torchvision.transforms.functional as F
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel

torch.multiprocessing.set_sharing_strategy('file_system')

class ToTensorWithoutScaling:
    """Convert PIL image or numpy array to a PyTorch tensor without scaling the values."""
    def __call__(self, pic):
        # Convert the PIL Image or numpy array to a tensor (without scaling).
        return F.pil_to_tensor(pic).long()

class RandomCropPair:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, target):
        # Ensure both img and target are PIL Images
        if not isinstance(img, Image.Image) or not isinstance(target, Image.Image):
            raise ValueError("Both img and target must be PIL Images")

        # Get random crop coordinates
        i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))

        # Apply the same crop to both the image and the target
        img = T.functional.crop(img, i, j, h, w)
        target = T.functional.crop(target, i, j, h, w)

        return img, target

def determine_lora_type(lora_dir):
    lora_types = ["stopgrad", "hybrid-reverse", "hybrid", "freeze"]
    for lora_type in lora_types:
        if lora_type in lora_dir:
            return lora_type
    return None

@hydra.main(config_path="configs", config_name="extract_embs.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    lora_dir = cfg.lora_dir
    if lora_dir == "None":
        lora_dir = None
    if_lora = True if lora_dir is not None else False
    save_imgs = cfg.save_imgs
    # seed_everything(seed=0, workers=True)

    load_size = 224
    if cfg.img_multiply > 1:
        load_size = 224 * cfg.img_multiply

    if cfg.model_type == "dinov2":
        final_size = 16
        kernel_size = 14
    else:
        final_size = 14
        kernel_size = 16

    transform = T.Compose([
        T.Resize(load_size, InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])
    

    target_transforms = T.Compose([
        T.Resize(load_size, InterpolationMode.NEAREST),
        T.CenterCrop(load_size),
        ToTensorWithoutScaling()
    ])

    dataset = get_dataset(
        # cfg.pytorch_data_dir,
        '/mnt/haiwen/datasets',
        cfg.dataset,
        "train",
        transform=transform,
        target_transform=target_transforms,
        include_labels=True,
        random_crop=False)

    loader = DataLoader(
        dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    
    # if cfg.dataset == "sa1b":
    #     val_loader = []
    # else:
    
    val_dataset = get_dataset(
            # cfg.pytorch_data_dir,
            '/mnt/haiwen/datasets',
            cfg.dataset,
            "val",
            transform=transform,
            target_transform=target_transforms,
            include_labels=True,
            random_crop=False)

    val_loader = DataLoader(
            val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    model, patch_size, dim = get_featurizer(cfg.model_type, cfg.activation_type, num_classes=1000)
    
    if lora_dir is not None:
        if cfg.model_type in ["vit", "dinov2"]:
            target_modules = ["qkv", "attn.proj"]
        elif cfg.model_type == "maskclip":
            target_modules = ["out_proj"]
        lora_config = LoraConfig(
                        r=4, # the dimension of the low-rank matrices
                        lora_alpha=16, # alpha/r is used to rescale the deltaW for updates. We don't tune this when tuning r to be able to use the same hyperparameters for different r.
                        # target_modules=["qkv", "proj"], # Use regex to match the module names. ["qkv", "proj"] belong to the self-attention layers. We don't tune the MLP for now.
                        target_modules=target_modules,
                        lora_dropout=0.,
                        bias="none", # lora_only, all
                        # task_type="FEATURE_EXTRACTION",
                        # modules_to_save=modules_to_save  # This argument serves for adding new tokens.
                    )
        model = get_peft_model(model, lora_config)

        ## Loading pretrained weights
        ckpt_weight = torch.load(lora_dir)['state_dict']
        lora_model_cpkt = {k: v for k, v in ckpt_weight.items() if 'model.0' in k}
        lora_model_cpkt = {k.replace('model.0.', ''): v for k, v in lora_model_cpkt.items()}
        model.load_state_dict(lora_model_cpkt)

        # PeftModel.from_pretrained(model, lora_dir)
    

    model = model.cuda()
    # Load data
    # Save the embeddings into a file "embeddings_{cfg.activation_type}.npz"
    # with keys "feats" and "labels"
    train_feats = []
    train_labels = []
    train_imgs = []
    for i, batch in tqdm(enumerate(loader)):
        if i == 25000: # 30000
            break
        img = batch["img"].cuda()
        label = batch["label"]
        feats = model(img)
        img = img.detach().cpu().numpy()
        feats = feats.detach().cpu().numpy()
        if save_imgs:
            train_imgs.append(img)
        train_feats.append(feats)
        train_labels.append(label)
        del img
        del label
        del batch
    # Save the embeddings into a file "embeddings_{cfg.activation_type}.npz"
    # with keys "feats" and "labels"
    if save_imgs:
        train_imgs = np.concatenate(train_imgs, axis=0)
    train_feats = np.concatenate(train_feats, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    os.makedirs(join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train"), exist_ok=True)
    if if_lora:
        lora_type = determine_lora_type(lora_dir)
        print("lora_type: ", lora_type)
        if lora_type is not None:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train",
                    f"embeddings_{cfg.task}_{cfg.model_type}_lora_{lora_type}.npz"),
                imgs=train_imgs, feats=train_feats, labels=train_labels)
        else:
            if "HR" in cfg.lora_dir:
                np.savez(
                    join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train",
                        f"embeddings_{cfg.task}_{cfg.model_type}_lora_hr.npz"),
                    imgs=train_imgs, feats=train_feats, labels=train_labels)
            else:
                np.savez(
                    join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train",
                        f"embeddings_{cfg.task}_{cfg.model_type}_lora.npz"),
                    imgs=train_imgs, feats=train_feats, labels=train_labels)
    else:
        if save_imgs:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train",
                    f"embeddings_{cfg.task}_{cfg.model_type}_{load_size}.npz"),
                    imgs=train_imgs, feats=train_feats, labels=train_labels)
        else:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "train",
                    f"embeddings_{cfg.task}_{cfg.model_type}_{load_size}.npz"),
                feats=train_feats, labels=train_labels)
    del train_feats
    del train_labels
    
    # val data
    val_imgs = []
    val_feats = []
    val_labels = []
    for i, batch in tqdm(enumerate(val_loader)):
        img = batch["img"].cuda()
        label = batch["label"]
        feats = model(img)
        img = img.detach().cpu().numpy()
        feats = feats.detach().cpu().numpy()
        if save_imgs:
            val_imgs.append(img)
        val_feats.append(feats)
        val_labels.append(label)
        del img
        del label
        del batch
    if save_imgs:
        val_imgs = np.concatenate(val_imgs, axis=0)
    val_feats = np.concatenate(val_feats, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    # Save the embeddings into a file "embeddings_{cfg.activation_type}.npz"
    # with keys "feats" and "labels"
    os.makedirs(join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val"), exist_ok=True)
    if if_lora:
        lora_type = determine_lora_type(lora_dir)
        if lora_type is not None:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val",
                    f"embeddings_{cfg.task}_{cfg.model_type}_lora_{lora_type}.npz"),
                imgs=val_imgs, feats=val_feats, labels=val_labels)
        else:
            if "HR" in cfg.lora_dir:
                np.savez(
                    join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val",
                        f"embeddings_{cfg.task}_{cfg.model_type}_lora_hr.npz"),
                    imgs=val_imgs, feats=val_feats, labels=val_labels)
            else:
                np.savez(
                    join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val",
                        f"embeddings_{cfg.task}_{cfg.model_type}_lora.npz"),
                    imgs=val_imgs, feats=val_feats, labels=val_labels)
    else:
        if save_imgs:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val",
                    f"embeddings_{cfg.task}_{cfg.model_type}_{load_size}.npz"),
                imgs=val_imgs, feats=val_feats, labels=val_labels)
        else:
            np.savez(
                join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type, "val",
                    f"embeddings_{cfg.task}_{cfg.model_type}_{load_size}.npz"),
                feats=val_feats, labels=val_labels)
    
    return None



if __name__ == "__main__":
    my_app()
