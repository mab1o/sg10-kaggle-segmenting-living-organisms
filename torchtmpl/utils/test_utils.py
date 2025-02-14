import logging
import os

# External imports
import torch
import yaml

from torchtmpl import data
from . import train_utils


def load_model_config(model_dir, model_config_name):
    """Load the model configuration file from the specified directory.

    Arguments:
        model_dir: Directory where the model configuration is stored.
        model_config_name: Name of the model configuration file.

    Returns:
        A dictionary containing the configuration parameters loaded from the YAML file.

    """
    config_path = os.path.join(model_dir, model_config_name)
    logging.debug(f"Loading model: {model_dir} with config: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def predict_and_insert(dataset, model, device, use_probs=False):
    """Predict masks or probabilities for each image in the dataset and inserts them into the dataset.

    Args:
        dataset: The dataset containing the images to predict on.
        model: The model to use for making predictions.
        device: The device (CPU or GPU) on which the model is running.
        use_probs: A boolean flag to decide whether to predict probabilities or binary masks (default is False).

    Returns:
        None: Predictions are inserted directly into the dataset.

    """
    with torch.inference_mode():
        for idx_img in range(dataset.image_patches[0][0] * dataset.image_patches[0][1]):
            if idx_img % 400 == 0:
                logging.info(
                    f"  - Predicting {'probabilities' if use_probs else 'masks'} for mask {idx_img}"
                )
            image = dataset[idx_img].unsqueeze(0).to(device)

            if use_probs:
                # Predict probabilities
                if "segmentation_models_pytorch" in type(model).__module__:
                    logits = model(image)
                    probs = torch.sigmoid(logits).half()
                else:
                    probs = model.predict_probs(image).half()
                dataset.insert(probs)
            else:
                # Predict binary masks
                dataset.insert(model.predict(image))


def load_dataset(config, model_config):
    """Load the test and optionally the training dataset based on the configuration.

    Args:
        config: A dictionary containing configuration settings, including data paths.
        model_config: The model configuration, which includes parameters like patch size.

    Returns:
        dataset_test: The test dataset.
        dataset_train: The training dataset (or None if not specified in the config).

    """
    dataset_test = data.PlanktonDataset(
        config["data"]["testpath"], model_config["data"]["patch_size"], mode="test"
    )
    logging.info(f"  - Number of test samples: {len(dataset_test)}")

    dataset_train = None
    if config["test"].get("use_train", False):
        dataset_train = data.PlanktonDataset(
            config["data"]["trainpath"], model_config["data"]["patch_size"]
        )
        logging.info(f"  - Number of train samples: {len(dataset_train)}")
    return dataset_test, dataset_train


def calc_and_show_proba(config, device, model_config, dataset_train, model):
    """Calculate probabilities for the training dataset and compares them to real masks.

    Args:
        config: A dictionary containing configuration settings, including data paths.
        device: The device (CPU or GPU) on which the model is running.
        model_config: The model configuration, which includes parameters like patch size.
        dataset_train: The training dataset used for predictions and evaluation.
        model: The model to use for predicting probabilities.

    """
    logging.info("= Predict probabilities and compare to real masks")
    dataset_train_proba = data.PlanktonDataset(
        config["data"]["trainpath"],
        model_config["data"]["patch_size"],
        mode="test",
    )
    train_utils.predict_and_insert(dataset_train_proba, model, device, use_probs=True)

    logging.info("= Show probabilities and compare to real masks")
    data.show_predicted_mask_proba_vs_real_mask_binary(
        dataset_train_proba, 0, dataset_train, "proba_compared_real_1.png"
    )


def more_eval(dataset_test, dataset_train):
    """Perform additional evaluations using both the test and training datasets.

    Args:
        dataset_test: The test dataset used for comparison of predicted masks.
        dataset_train: The training dataset used for additional evaluation (e.g., image reconstruction).

    """
    logging.info("= Extra evaluation using training data")
    data.show_image_mask_from(dataset_train, 0, "image_reconstruct_1.png")
    data.show_mask_predict_compare_to_real(
        dataset_test, 0, dataset_train, "compare_mask_1.png"
    )
