import logging
import os

import torch
import torch.nn
import yaml

from torchtmpl import models


def load_configs(models_dir, model_config_name):
    """Load model configuration files from the specified directories.

    Args:
        models_dir (list): List of directories where the model configuration files are located.
        model_config_name (str): Name of the configuration file for each model.

    Returns:
        models_config (list): List of dictionaries containing the model configurations.
        patch_sizes (list): List of patch sizes (tuples) for each model.

    Raises:
        ValueError: If patch sizes are inconsistent across models.

    """
    models_config = []
    patch_sizes = []
    for model_dir in models_dir:
        config_path = os.path.join(model_dir, model_config_name)
        logging.debug(f"Loading model: {model_dir} with config: {config_path}")

        with open(config_path) as f:
            model_config = yaml.safe_load(f)
        models_config.append(model_config)

        patch_size = model_config["data"]["patch_size"]
        patch_sizes.append(tuple(patch_size))

    # Check all models have same patch size
    if len(set(patch_sizes)) == 1:
        logging.info(f"All models have the same patch size: {patch_sizes[0]}")
    else:
        logging.error("Patch sizes are not consistent across models.")
        raise ValueError("Inconsistent patch sizes across models.")
    return models_config, patch_sizes


def find_models_weight_path(model_weight_name, models_dir):
    """Find the paths to the model weight files in the provided directories.

    Args:
        model_weight_name (str): Name of the model weight file to search for.
        models_dir (list): List of directories to search for the model weight files.

    Return:
        models_weight_path (list): List of paths where the model weight files are located.

    Raise:
        ValueError: If no weight files are found in the specified directories.

    """
    models_weight_path = [
        os.path.join(model_dir, model_weight_name)
        for model_dir in models_dir
        if os.path.exists(os.path.join(model_dir, model_weight_name))
    ]
    logging.info(f"Found {len(models_weight_path)} models in {models_dir}")

    if not models_weight_path:
        logging.error("No models found!")
        raise ValueError("No models fund")
    return models_weight_path


def load_models(
    device, use_tta, models_config, input_size, num_classes, models_weight_path
):
    """Load models, optionally applying Test-Time Augmentation (TTA).

    Args:
        device (torch.device): Device (CPU or GPU) on which to load the models.
        use_tta (bool): Whether to apply Test-Time Augmentation (TTA) to the models.
        models_config (list): List of dictionaries containing model configurations.
        input_size (tuple): The input size (height, width) for the models.
        num_classes (int): The number of output classes for the models.
        models_weight_path (list): List of paths to the model weight files.

    Return:
        models (list): List of loaded models (with or without TTA applied).

    Raise:
        ValueError: If there is a mismatch in the number of models and weight paths.

    """
    if len(models_weight_path) != len(models_config):
        raise ValueError("Mismatch size between configs and models weight")

    models = []
    for model_weight_path, model_config in zip(models_weight_path, models_config):
        model = build_and_load_model(
            model_config["model"], input_size, num_classes, model_weight_path, device
        )
        if use_tta:
            model = apply_tta(model, use_tta)
        models.append(model)

    if not models:
        logging.error("No valid models could be loaded!")
        raise ValueError("0 Models Loaded")
    logging.debug(f"Loaded {len(models)} models successfully!")
    return models


def predict_masks(device, dataset_test, models):
    """Predict masks for the test dataset using the provided models.

    Args:
        device (torch.device): Device (CPU or GPU) for prediction.
        dataset_test (Dataset): The test dataset to generate predictions for.
        models (list): List of models to use for making predictions.

    """
    for image_idx, (num_patches_x, num_patches_y) in enumerate(
        dataset_test.image_patches
    ):
        predict_patchs(
            device, dataset_test, models, image_idx, num_patches_x, num_patches_y
        )


def predict_patchs(
    device, dataset_test, models, image_idx, num_patches_x, num_patches_y
):
    """Predict the patches for a specific image using the provided models.

    Args:
        device (torch.device): Device (CPU or GPU) for prediction.
        dataset_test (Dataset): The test dataset containing the image to predict.
        models (list): List of models to use for predicting the image patches.
        image_idx (int): The index of the image in the dataset.
        num_patches_x (int): The number of patches along the x-axis.
        num_patches_y (int): The number of patches along the y-axis.

    """
    logging.info(f"Predicting patches for image {image_idx}")

    for idx_patch in range(num_patches_x * num_patches_y):
        if idx_patch % 100 == 0:
            logging.info(f"Predict patch {idx_patch} / {num_patches_x * num_patches_y}")

        global_idx = idx_patch + sum(
            x * y for x, y in dataset_test.image_patches[:image_idx]
        )  # Calcul de l'index global

        image = dataset_test[global_idx].unsqueeze(0).to(device)

        # Moyenne des prédictions de tous les modèles
        with torch.inference_mode():
            predictions = [torch.sigmoid(model(image)) for model in models]
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)

        # Appliquer un seuil pour la segmentation binaire
        binary_prediction = (avg_prediction > 0.5).long()
        dataset_test.insert(binary_prediction)


def build_and_load_model(model_config, input_size, num_classes, model_path, device):
    """Build and load a model from the given configuration and weight path.

    Args:
        model_config (dict): The configuration dictionary for the model.
        input_size (tuple): The input size (height, width) for the model.
        num_classes (int): The number of output classes for the model.
        model_path (str): Path to the model's weight file.
        device (torch.device): The device (CPU or GPU) on which to load the model.

    Returns:
        model (torch.nn.Module): The built and loaded model.

    """
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def apply_tta(model, use_tta):
    """Apply Test-Time Augmentation (TTA) to the model if requested.

    Args:
        model (torch.nn.Module): The model to apply TTA to.
        use_tta (bool): Whether to apply TTA.

    Returns:
        model (torch.nn.Module): The model with TTA applied if `use_tta` is True.

    """
    import ttach as tta

    if use_tta:
        logging.info("Test-Time Augmentation (TTA) ACTIVÉ")
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
        return tta.SegmentationTTAWrapper(model, tta_transforms)
    else:
        logging.info("Test-Time Augmentation (TTA) DÉSACTIVÉ")
        return model
