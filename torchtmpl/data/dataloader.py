# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.utils.data
import matplotlib.pyplot as plt

# Local imports
from . import planktonds


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def get_dataloaders(data_config, use_cuda):
    """
    Prépare les DataLoaders pour entraîner et valider un modèle avec un dataset PlanktonDataset.

    Args:
        data_config (dict): Configuration du dataset contenant :
            - "valid_ratio" (float): Proportion des données pour validation.
            - "batch_size" (int): Taille du batch.
            - "num_workers" (int): Nombre de workers pour DataLoader.
            - "trainpath" (str): Chemin vers le répertoire contenant les images et masques.
            - "patch_size" ([int, int]): Taille des patchs à extraire.
        use_cuda (bool): Indique si CUDA est utilisé.
        plankton_dataset_cls (type): Classe du dataset (ex : PlanktonDataset).

    Returns:
        tuple: Contient les DataLoaders pour train et validation, la taille d'entrée et le nombre de classes (C, H, W).
    """
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation (PlanktonDataset)")

    base_dataset = planktonds.PlanktonDataset(
        image_mask_dir=data_config['trainpath'],
        patch_size=data_config["patch_size"],
        mode="train"
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = 1
    input_size = tuple(base_dataset[0][0].shape)
        
    logging.info(f"  - Input size is {input_size} and the number of classe is {num_classes}.")
    return train_loader, valid_loader, input_size, num_classes
