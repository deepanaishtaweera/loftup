import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import os
import glob
import time

import hydra
from omegaconf import DictConfig
import rerun as rr

# --- Assumed imports from your project ---
# Make sure these files are in your project directory.
from featurizers import get_featurizer
from upsamplers import get_upsampler, load_upsampler_weights, norm
from utils import pca, COCOSTUFF_27_CATEGORIES


def setup_rerun(cfg: DictConfig):
    """Initializes and configures the Rerun visualizer."""
    # Create a unique name for the Rerun log file
    run_name = f"{cfg.model_type}_{cfg.upsampler_type}_eval"
    current_time = time.strftime("%Y%m%d_%H%M%S")

    rr.init(run_name, spawn=False)  # spawn=False to control saving manually

    # Define where to save the .rrd file
    output_path = os.path.join(cfg.output_root, f"{run_name}_{current_time}.rrd")
    os.makedirs(cfg.output_root, exist_ok=True)
    rr.save(output_path)

    print(f"Rerun is active. Saving log to: {output_path}")


def get_frame_list(frame_ranges):
    frames = []
    for range_dict in frame_ranges:
        start = range_dict["start"]
        end = range_dict["end"]
        step = range_dict["step"]
        frames.extend(range(start, end + 1, step))
    return sorted(list(set(frames)))  # Remove duplicates and sort


@hydra.main(config_path="configs", config_name="eval_seg_trail.yaml", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    Main function to run evaluation and log results to Rerun.
    """
    setup_rerun(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. --- Load Models ---
    print("Loading models...")
    # Load the feature extractor (e.g., DINOv2)
    featurizer, _, feat_dim = get_featurizer(cfg.model_type)
    featurizer.to(device)
    featurizer.eval()

    # Load the upsampler
    upsampler = get_upsampler(cfg.upsampler_type, feat_dim, cfg=cfg)
    if cfg.upsampler_path and cfg.upsampler_type != "bilinear":
        upsampler = load_upsampler_weights(upsampler, cfg.upsampler_path, feat_dim)
    if upsampler:
        upsampler.to(device)
        upsampler.eval()

    # Load the trained linear classifier
    n_classes = len(COCOSTUFF_27_CATEGORIES)
    classifier = torch.nn.Conv2d(feat_dim, n_classes, 1)

    try:
        state_dict = torch.load(cfg.classifier_checkpoint, map_location=device)
        classifier.load_state_dict(state_dict)
        print(f"Successfully loaded classifier from {cfg.classifier_checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    classifier.to(device)
    classifier.eval()

    # 2. --- Define Image Transforms ---
    transform = T.Compose(
        [
            T.Resize(cfg.load_size, InterpolationMode.BILINEAR),
            T.CenterCrop(cfg.load_size),
            T.ToTensor(),
            norm,  # The normalization from your utils.py
        ]
    )

    # 3. --- Setup Rerun Annotation Context for Segmentation ---
    # This tells Rerun how to label and color the classes
    labels_mapping = [
        rr.AnnotationInfo(id=cat["id"], label=cat["name"], color=cat["color"])
        for cat in COCOSTUFF_27_CATEGORIES
    ]
    rr.log("eval", rr.AnnotationContext(labels_mapping), static=True)

    # 4. --- Process Images ---
    frame_list = get_frame_list([{"start": 400, "end": 10000, "step": 10}])
    image_paths = [os.path.join(cfg.image_dir, f"{frame:05d}.png") for frame in frame_list]
    print(f"Found {len(image_paths)} images in {cfg.image_dir}")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                print(
                    f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}"
                )

                # Set the current "frame" in the Rerun timeline
                rr.set_time(timeline="frame", sequence=i)

                # Load and preprocess the image
                raw_image = Image.open(img_path).convert("RGB")
                img_tensor = transform(raw_image).unsqueeze(0).to(device)

                # --- Inference ---
                lr_features = featurizer(img_tensor)

                if upsampler:
                    guidance_image = F.interpolate(
                        img_tensor,
                        size=(cfg.guidance_res, cfg.guidance_res),
                        mode="bilinear",
                    )
                    hr_features = upsampler(lr_features, guidance_image)
                else:
                    hr_features = lr_features

                logits = classifier(hr_features)

                prediction = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                # --- Rerun Logging ---
                # Log the original image
                rr.log("eval/original_image", rr.Image(np.array(raw_image)))

                # Visualize features with PCA
                # Note: `pca` must be available in your utils.py
                [red_lr_feats], fit_pca = pca([lr_features[0].unsqueeze(0)])
                [red_hr_feats], _ = pca([hr_features[0].unsqueeze(0)], fit_pca=fit_pca)

                rr.log(
                    "eval/pca_features_low_res",
                    rr.Image(red_lr_feats[0].cpu().numpy().transpose(1, 2, 0)),
                )
                rr.log(
                    "eval/pca_features_high_res",
                    rr.Image(red_hr_feats[0].cpu().numpy().transpose(1, 2, 0)),
                )

                # Log the final segmentation prediction
                rr.log("eval/prediction", rr.SegmentationImage(prediction))

            except Exception as e:
                print(f"Could not process {img_path}. Error: {e}")

    print(f"\nEvaluation complete. Open the .rrd file with the Rerun viewer.")
    print(f"Command: rerun <path_to_rrd_file>")


if __name__ == "__main__":
    evaluate()
