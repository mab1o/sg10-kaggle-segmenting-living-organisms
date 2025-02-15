import torch
import os
import logging
import yaml
import pathlib


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    path="checkpoint.pth",
):
    """Save the current model state,..

    Args:
        model: The model to save the state of.
        optimizer: The optimizer to save the state of.
        scheduler: The scheduler to save the state of (can be None).
        epoch: The current epoch number.
        loss: The current loss value.
        path: The path where the checkpoint file will be saved.

    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }
    torch.save(checkpoint, path)


def save_datasets_indice(train_loader, val_loader, path="dataset"):
    """Save the indices of the training and validation datasets.

    Args:
        train_loader: The training data loader.
        val_loader: The validation data loader.
        path: The path where the dataset indices will be saved (default: "dataset").

    """
    checkpoint = {
        "train_loader_state": train_loader.dataset.indices
        if hasattr(train_loader.dataset, "indices")
        else None,
        "val_loader_state": val_loader.dataset.indices
        if hasattr(val_loader.dataset, "indices")
        else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    path_checkpoint,
    path_dataset,
    device,
):
    """Load a saved checkpoint.

    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        scheduler: The scheduler to load the state into (can be None).
        train_loader: The training data loader.
        valid_loader: The validation data loader.
        path_checkpoint: The path to the checkpoint file.
        path_dataset: The path to the dataset file containing dataset indices.
        device: The device (CPU/GPU) where the model will be loaded.

    Returns:
        epoch: The epoch to resume from.
        loss: The last recorded loss value.

    """
    if not os.path.exists(path_checkpoint):
        logging.warning(
            f"No checkpoint found at {path_checkpoint}, starting from scratch."
        )
        return 0, None

    checkpoint = torch.load(path_checkpoint, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    if not os.path.exists(path_dataset):
        logging.warning(
            f"No checkpoint found at {path_dataset}, starting from scratch."
        )
        return 0, None

    datasets_indices = torch.load(path_dataset, map_location=device)

    if datasets_indices.get("train_loader_state") is not None:
        train_loader.dataset.indices = datasets_indices["train_loader_state"]
    if datasets_indices.get("val_loader_state") is not None:
        valid_loader.dataset.indices = datasets_indices["val_loader_state"]

    logging.info(
        f"Checkpoint loaded from {path_dataset} and {path_checkpoint}, resuming from epoch {epoch}"
    )
    return epoch, loss


def update_config_train(config):
    """Load and return a training configuration file if a previous model found.

    Args:
        config: The configuration dictionary, which may contain information
            about previous training.

    Returns:
        first_train: Boolean indicating if this is the first training.
        config: The updated configuration dictionary.
        logdir: The directory path where the logs or model directory is stored (if resuming training).

    """
    first_train = config.get("train_from", "") == ""

    if first_train:
        return first_train, config, None

    logdir = config["train_from"]["model_dir"]
    config_path = os.path.join(logdir, config["train_from"]["model_config_name"])
    logging.debug(f"Loading config: {config_path}")

    with open(config_path) as f:
        return first_train, yaml.safe_load(f), pathlib.Path(logdir)
