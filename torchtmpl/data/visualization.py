# External imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
from sklearn.metrics import f1_score
from PIL import Image

# Local imports
from . import patch
from . import planktonds

Image.MAX_IMAGE_PIXELS = None  # désactive la limite de taille


def _show_image_mask_given(img, mask, image_name="plankton_sample.png"):
    """
    Display an image and its mask side by side.
    Args:
        img (np.ndarray or torch.Tensor): Image array, expected shape (H, W, 1) or (H, W).
        mask (np.ndarray or torch.Tensor): Mask array, expected shape (H, W).
        image_name (str): File name to save the resulting figure.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, interpolation="none", cmap="tab20c")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(image_name, bbox_inches="tight", dpi=300)
    logging.info(f"  - Saved visualization to {image_name}")


def _show_mask_given(predict_mask, image_name="compare_mask.png"):
    """
    Display a real mask and its prediction side by side.
    Args:
        predict_mask (np.ndarray or torch.Tensor): Predicted mask, expected shape (H, W).
        image_name (str): File name to save the resulting figure.
    """
    plt.imshow(predict_mask, interpolation="none", cmap="tab20c")
    plt.title("Predict Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(image_name, bbox_inches="tight", dpi=300)
    logging.info(f"  - Saved mask comparison to {image_name}")


def show_image_mask_from(
    ds: planktonds.PlanktonDataset, idx, image_name="plankton_sample.png"
):
    """
    Show the complete image and its reconstructed mask.
    Args:
        ds (PlanktonDataset): Dataset object.
        idx (int): Index of the image to display.
        image_name (str): File name to save the resulting figure.
    """
    assert 0 <= idx < len(ds.image_files), (
        f"Index {idx} out of range. Dataset has {len(ds.image_files)} images."
    )

    image_path = os.path.join(ds.image_mask_dir, ds.image_files[idx])
    img = patch.extract_patch_from_ppm(image_path, 0, 0, ds.images_size[idx])
    mask = ds.reconstruct_mask(idx)

    _show_image_mask_given(img, mask, image_name)


def show_mask_predict_compare_to_real(
    ds: planktonds.PlanktonDataset, idx, real_dataset, image_name="compare_mask.png"
):
    """
    Compare the real mask and predicted mask for an image.
    Args:
        ds (PlanktonDataset): Dataset object with predictions.
        idx (int): Index of the image.
        real_dataset (PlanktonDataset): Dataset with real masks.
        image_name (str): File name to save the resulting figure.
    """
    assert 0 <= idx < len(ds.image_files), (
        f"Index {idx} out of range. Dataset has {len(ds.image_files)} images."
    )
    assert isinstance(real_dataset, planktonds.PlanktonDataset), (
        "real_dataset must be an instance of PlanktonDataset."
    )

    real_mask_path = os.path.join(
        real_dataset.image_mask_dir, real_dataset.mask_files[idx]
    )
    real_mask = patch.extract_patch_from_ppm(
        real_mask_path, 0, 0, real_dataset.images_size[idx]
    )
    real_mask = np.where(real_mask < 8, 0, 1)

    predict_mask = ds.reconstruct_mask(idx)
    predict_mask += real_mask

    _show_mask_given(predict_mask, image_name)


def show_plankton_patch_image(
    ds: planktonds.PlanktonDataset, idx, image_name="plankton_patch.png"
):
    """
    Display a patch of the plankton image at a specific index.

    Args:
        ds (PlanktonDataset): Dataset object.
        idx (int): Index of the patch to display.
        image_name (str): File name to save the resulting figure.
    """
    if ds.mode == "train":
        img, mask = ds[idx]
    else:
        img = ds[idx]
        mask = ds.mask_files[idx] if idx < len(ds.mask_files) else None

    _show_image_mask_given(img, mask, image_name)


def show_tensor_image_given(X):
    """
    Display a single image.
    Args:
        X (torch.Tensor or np.ndarray): Image tensor or array, shape (C, H, W) or (H, W).
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    plt.figure()
    if X.ndim == 3:  # (C, H, W)
        plt.imshow(X[0] if X.shape[0] == 1 else X.transpose(1, 2, 0))
    else:  # (H, W)
        plt.imshow(X, cmap="gray")
    plt.axis("off")
    plt.show()


def show_predicted_mask_proba_vs_real_mask_binary(
    ds: planktonds.PlanktonDataset,
    idx: int,
    real_dataset: planktonds.PlanktonDataset,
    image_name: str = "proba_compared_real.png",
):
    """
    Compare the predicted probability heatmap with the real mask for an image,
    and estimate the best threshold (for F1) using a ternary search across [0,1].
    We do 5 iterations, and only display the heatmap + real mask side by side.
    """

    assert 0 <= idx < len(ds.image_files), f"Index {idx} out of range."
    assert isinstance(real_dataset, planktonds.PlanktonDataset), (
        "real_dataset must be an instance of PlanktonDataset."
    )

    # 1) Load the real mask
    real_mask_path = os.path.join(
        real_dataset.image_mask_dir, real_dataset.mask_files[idx]
    )
    real_mask = patch.extract_patch_from_ppm(
        real_mask_path, 0, 0, real_dataset.images_size[idx]
    )
    real_mask = np.where(real_mask < 8, 0, 1).astype(np.uint8)

    # 2) Load predicted probabilities
    proba_mask = ds.reconstruct_mask(idx, binary=False)

    # Helper to compute F1 at a given threshold
    def f1_at_threshold(th):
        bin_mask = (proba_mask > th).to(torch.uint8)
        return f1_score(
            real_mask.flatten(), bin_mask.cpu().numpy().flatten(), zero_division=1
        )

    # Ternary search is faster than gridsearch.
    # 3) Ternary search for threshold in [low, high] with 5 iterations
    low, high = 0.0, 1.0
    for _ in range(5):
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        f1_1 = f1_at_threshold(mid1)
        f1_2 = f1_at_threshold(mid2)

        if f1_1 < f1_2:
            # Best is in [mid1, high]
            low = mid1
        else:
            # Best is in [low, mid2]
            high = mid2

    # After 5 iterations, pick the midpoint of [low, high]
    best_threshold = (low + high) / 2
    best_f1 = f1_at_threshold(best_threshold)

    logging.info(
        f"[Ternary Search] Best threshold ~ {best_threshold:.4f}, F1 = {best_f1:.4f}"
    )

    f1_threshold_0_5 = f1_at_threshold(0.5)
    logging.info(f"Default 0.5 threshold, F1 = {f1_threshold_0_5:.4f}")

    # 4) Plot only probability heatmap + real mask side by side
    plt.figure(figsize=(10, 5))

    # (a) Predicted mask (Probability heatmap)
    plt.subplot(1, 2, 1)
    plt.imshow(proba_mask, cmap="viridis")
    plt.title("Predicted Probabilities (Heatmap)")
    plt.colorbar()
    plt.axis("off")

    # (b) Real mask (binary)
    plt.subplot(1, 2, 2)
    plt.imshow(real_mask, cmap="tab20c")
    plt.title("Real Mask")
    plt.axis("off")

    # Add the best threshold as a title or annotation
    plt.suptitle(
        f"Thresholds: Best = {best_threshold:.2f} (F1 = {best_f1:.4f}), 0.5 (F1 = {f1_threshold_0_5:.4f})",
        fontsize=12,
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(image_name, bbox_inches="tight", dpi=300)
    logging.info(f"Saved probability vs real mask comparison to {image_name}")


def show_validation_image_vs_predicted_mask(
    ds: planktonds.PlanktonDataset,
    idx: int,
    validation_dataset: planktonds.PlanktonDataset,
    image_name: str = "validation_vs_predicted.png",
):
    """
    Compare the validation image with the predicted mask (probability heatmap) for a given index.

    Args:
        ds (PlanktonDataset): Dataset object containing the predicted probabilities.
        idx (int): Index of the image to compare.
        validation_dataset (PlanktonDataset): Dataset object containing the validation images.
        image_name (str): File name to save the resulting figure.
    """

    # 1) Load the validation image
    val_image_name = validation_dataset.image_files[idx]
    print(f"Selected image = {val_image_name}")

    val_image_path = os.path.join(
        validation_dataset.image_mask_dir, validation_dataset.image_files[idx]
    )

    val_image = plt.imread(val_image_path)

    # 2) Load the predicted mask (probability heatmap)
    proba_mask = ds.reconstruct_mask(idx, binary=False)

    # 3) Plot the validation image side by side with the predicted mask
    plt.figure(figsize=(10, 5))

    # (a) Validation image
    plt.subplot(1, 2, 1)
    plt.imshow(val_image, cmap="gray")
    plt.title("Validation Image")
    plt.axis("off")

    # (b) Predicted probability heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(proba_mask, cmap="viridis")
    plt.title("Predicted Probabilities (Heatmap)")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(image_name, bbox_inches="tight", dpi=300)
    logging.info(f"Saved validation vs predicted comparison to {image_name}")
