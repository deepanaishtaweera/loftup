import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import os
import glob
import time
import math

import hydra
from omegaconf import DictConfig
import rerun as rr

# --- Assumed imports from your project ---
# Make sure these files are in your project directory.
from featurizers import get_featurizer
from upsamplers import get_upsampler, load_upsampler_weights, norm
from utils import pca, COCOSTUFF_27_CATEGORIES, compute_traversability_from_logits


def run_model_sliding_window(
    model,
    upsampler,
    classifier,
    img_path,
    crop_size=224,
    stride_rate=2 / 3,
    batch_size=8,
    max_image_size=None,
    device="cuda",
):
    """
    Runs a model on a high-resolution image by processing it in overlapping
    patches and averaging the results, with memory optimization.

    Args:
        model: The feature extraction model (e.g., ViT).
        upsampler: The model used to upsample low-resolution features.
        classifier: The classification head for segmentation.
        img_path (str): Path to the high-resolution input image.
        crop_size (int): The input size required by the model.
        stride_rate (float): The overlap between patches. 2/3 means 1/3 overlap.
        batch_size (int): The number of patches to process in a single batch to control VRAM usage.
        max_image_size (int, optional): If set, resizes the image's longest side to this value
                                        before processing. Defaults to None.
        device (str): The device to run the model on.

    Returns:
        torch.Tensor: The final high-resolution segmentation logits for the entire image.
        torch.Tensor: The normalized high-resolution image tensor.
    """
    # 1. Load and normalize the high-resolution image
    transform = T.Compose([T.ToTensor(), norm])

    img = Image.open(img_path).convert("RGB")

    # --- NEW: Optional image resizing to reduce total number of patches ---
    if max_image_size is not None:
        original_size = img.size
        if max(original_size) > max_image_size:
            # Calculate new size while preserving aspect ratio
            if original_size[0] > original_size[1]:  # Landscape
                new_w = max_image_size
                new_h = int(max_image_size * original_size[1] / original_size[0])
            else:  # Portrait or square
                new_h = max_image_size
                new_w = int(max_image_size * original_size[0] / original_size[1])

            print(f"Image resized from {original_size} to ({new_w}, {new_h})")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_tensor = transform(img).unsqueeze(0).to(device)
    _, _, h, w = img_tensor.shape

    # 2. Calculate patch grid and stride
    stride = int(crop_size * stride_rate)
    h_grids = int(math.ceil(1.0 * (h - crop_size) / stride)) + 1
    w_grids = int(math.ceil(1.0 * (w - crop_size) / stride)) + 1

    # 3. Create lists to hold patches and their positions
    crops = []
    positions = []
    for i in range(h_grids):
        for j in range(w_grids):
            h0 = i * stride
            w0 = j * stride
            h1 = min(h0 + crop_size, h)
            w1 = min(w0 + crop_size, w)

            crop = img_tensor[:, :, h0:h1, w0:w1]
            padded_crop = F.pad(
                crop, (0, crop_size - (w1 - w0), 0, crop_size - (h1 - h0))
            )

            crops.append(padded_crop)
            positions.append((h0, h1, w0, w1))

    print(f"Processing {len(crops)} patches with grid size: {h_grids}x{w_grids}")

    # 4. Process crops in mini-batches to conserve memory
    logits_list = []
    hr_feats_list = []
    for i in range(0, len(crops), batch_size):
        # Get a mini-batch of crops
        batch_crops = torch.cat(crops[i : i + batch_size], dim=0)

        with torch.no_grad():
            lr_feats_batch = model(batch_crops)
            hr_feats_batch = upsampler(lr_feats_batch, batch_crops)
            logits_batch = classifier(hr_feats_batch)

        # Move results to CPU to free up VRAM for the next batch
        logits_list.append(logits_batch.cpu())
        hr_feats_list.append(hr_feats_batch.cpu())
    # Concatenate all results from mini-batches and move back to target device
    logits_batch = torch.cat(logits_list, dim=0).to(device)
    hr_feats_batch = torch.cat(hr_feats_list, dim=0).to(device)

    # 5. Stitch the results back together by averaging overlaps
    n_classes = logits_batch.shape[1]
    final_logits = torch.zeros(1, n_classes, h, w, device=device)
    final_hr_feats = torch.zeros(1, hr_feats_batch.shape[1], h, w, device=device)
    count_norm = torch.zeros(1, 1, h, w, device=device)

    for i, (h0, h1, w0, w1) in enumerate(positions):
        logits_patch = logits_batch[i].unsqueeze(0)
        unpadded_logits_patch = logits_patch[:, :, : h1 - h0, : w1 - w0]
        final_logits[:, :, h0:h1, w0:w1] += unpadded_logits_patch

        # Handle high-resolution features with proper unpadding
        hr_feat_patch = hr_feats_batch[i].unsqueeze(0)
        unpadded_hr_feat_patch = hr_feat_patch[:, :, : h1 - h0, : w1 - w0]
        final_hr_feats[:, :, h0:h1, w0:w1] += unpadded_hr_feat_patch

        count_norm[:, :, h0:h1, w0:w1] += 1

    final_logits /= count_norm + 1e-8
    final_hr_feats /= count_norm + 1e-8
    return final_logits, img_tensor.squeeze(0), final_hr_feats


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

    # 2. --- Setup Rerun Annotation Context for Segmentation ---
    # This tells Rerun how to label and color the classes
    labels_mapping = [
        rr.AnnotationInfo(id=cat["id"], label=cat["name"], color=cat["color"])
        for cat in COCOSTUFF_27_CATEGORIES
    ]
    rr.log("eval", rr.AnnotationContext(labels_mapping), static=True)

    # 3. --- Process Images ---
    frame_list = get_frame_list([{"start": 400, "end": 10000, "step": 10}])
    image_paths = [
        os.path.join(cfg.image_dir, f"{frame:05d}.png") for frame in frame_list
    ]
    print(f"Found {len(image_paths)} images in {cfg.image_dir}")

    # Check if sliding window should be used
    use_sliding_window = getattr(cfg, "use_sliding_window", False)

    if use_sliding_window:
        print("Using sliding window approach to maintain original image resolution")
        # Sliding window parameters
        crop_size = cfg.load_size if hasattr(cfg, "load_size") else 224
        stride_rate = getattr(cfg, "stride_rate", 2 / 3)
        batch_size = getattr(cfg, "batch_size", 8)
        max_image_size = getattr(cfg, "max_image_size", None)
    else:
        print("Using standard resize/crop approach")
        # Define standard transforms for non-sliding window approach
        transform = T.Compose(
            [
                T.Resize(cfg.load_size, InterpolationMode.BILINEAR),
                T.CenterCrop(cfg.load_size),
                T.ToTensor(),
                norm,  # The normalization from your utils.py
            ]
        )

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                print(
                    f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}"
                )

                # Set the current "frame" in the Rerun timeline
                rr.set_time(timeline="frame", sequence=i)

                # Load the original image for logging
                raw_image = Image.open(img_path).convert("RGB")

                if use_sliding_window:
                    # --- Inference using sliding window ---
                    logits, img_tensor, hr_feats = run_model_sliding_window(
                        model=featurizer,
                        upsampler=upsampler,
                        classifier=classifier,
                        img_path=img_path,
                        crop_size=crop_size,
                        stride_rate=stride_rate,
                        batch_size=batch_size,
                        max_image_size=max_image_size,
                        device=device,
                    )

                    prediction = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

                    # Compute traversability prediction
                    for temperature in [0.5, 1.0, 2.0]:
                        traversability_pred = compute_traversability_from_logits(
                            logits, temperature=temperature
                        )
                        traversability_pred_np = traversability_pred.cpu().numpy()
                        # Log the traversability prediction
                        rr.log(
                            f"eval/traversability_{temperature}",
                            rr.DepthImage(traversability_pred_np),
                        )

                    # --- Rerun Logging ---
                    # Log the original image
                    rr.log(
                        "eval/original_image",
                        rr.Image(img_tensor.cpu().numpy().transpose(1, 2, 0)),
                    )

                    # Log the high-resolution features
                    # pca
                    [red_hr_feats], fit_pca = pca([hr_feats[0].unsqueeze(0)])
                    rr.log(
                        "eval/pca_features_high_res",
                        rr.Image(red_hr_feats[0].cpu().numpy().transpose(1, 2, 0)),
                    )

                    # Log the final segmentation prediction (now at original image resolution)
                    rr.log("eval/prediction", rr.SegmentationImage(prediction))

                else:
                    # --- Standard inference approach ---
                    img_tensor = transform(raw_image).unsqueeze(0).to(device)

                    # Inference
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

                    # Compute traversability prediction
                    traversability_pred = compute_traversability_from_logits(logits)
                    traversability_pred_np = traversability_pred.cpu().numpy()

                    # --- Rerun Logging ---
                    # Log the original image
                    rr.log("eval/original_image", rr.Image(np.array(raw_image)))

                    # Visualize features with PCA
                    [red_lr_feats], fit_pca = pca([lr_features[0].unsqueeze(0)])
                    [red_hr_feats], _ = pca(
                        [hr_features[0].unsqueeze(0)], fit_pca=fit_pca
                    )

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

                    # Log the traversability prediction
                    rr.log("eval/traversability", rr.Image(traversability_pred_np))

            except Exception as e:
                print(f"Could not process {img_path}. Error: {e}")

    print(f"\nEvaluation complete. Open the .rrd file with the Rerun viewer.")
    print(f"Command: rerun <path_to_rrd_file>")


if __name__ == "__main__":
    evaluate()
