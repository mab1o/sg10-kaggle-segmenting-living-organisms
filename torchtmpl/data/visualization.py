# External imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging

# Local imports
from . import patch
from . import planktonds

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


def show_image_mask_from(ds: planktonds.PlanktonDataset, idx, image_name="plankton_sample.png"):
    """
    Show the complete image and its reconstructed mask.
    Args:
        ds (PlanktonDataset): Dataset object.
        idx (int): Index of the image to display.
        image_name (str): File name to save the resulting figure.
    """
    assert 0 <= idx < len(ds.image_files), f"Index {idx} out of range. Dataset has {len(ds.image_files)} images."

    image_path = os.path.join(ds.image_mask_dir, ds.image_files[idx])
    img = patch.extract_patch_from_ppm(image_path, 0, 0, ds.images_size[idx])
    mask = ds.reconstruct_mask(idx)

    _show_image_mask_given(img, mask, image_name)


def show_mask_predict_compare_to_real(ds: planktonds.PlanktonDataset, idx, real_dataset, image_name="compare_mask.png"):
    """
    Compare the real mask and predicted mask for an image.
    Args:
        ds (PlanktonDataset): Dataset object with predictions.
        idx (int): Index of the image.
        real_dataset (PlanktonDataset): Dataset with real masks.
        image_name (str): File name to save the resulting figure.
    """
    assert 0 <= idx < len(ds.image_files), f"Index {idx} out of range. Dataset has {len(ds.image_files)} images."
    assert isinstance(real_dataset, planktonds.PlanktonDataset), "real_dataset must be an instance of PlanktonDataset."

    real_mask_path = os.path.join(real_dataset.image_mask_dir, real_dataset.mask_files[idx])
    real_mask = patch.extract_patch_from_ppm(real_mask_path, 0, 0, real_dataset.images_size[idx])
    real_mask = np.where(real_mask < 8, 0, 1)

    predict_mask = ds.reconstruct_mask(idx)
    predict_mask += real_mask

    _show_mask_given(predict_mask, image_name)


def show_plankton_patch_image(ds: planktonds.PlanktonDataset, idx, image_name="plankton_patch.png"):
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
