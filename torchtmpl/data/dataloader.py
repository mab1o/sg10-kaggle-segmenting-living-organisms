# Standard imports
import logging
import random

# External imports
import torch
import torch.utils.data
import albumentations as A

# Local imports
from . import planktonds

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
    quick_test = data_config["quick_test"]

    logging.info("  - Dataset creation (PlanktonDataset)")
    """
    # Définir les transformations pour l'entraînement
    train_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.02), rotate=(-15, 15), p=0.5),  # Remplace RandomRotate90
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        A.GaussianBlur(blur_limit=1, p=0.2),  # Ajout pour défocaliser légèrement
        A.MotionBlur(blur_limit=3, p=0.15),  # Remis pour imiter des flous naturels des scanners
        A.GaussianBlur(blur_limit=1, p=0.2),  # Léger flou pour éviter de surinterpréter le bruit
        A.ElasticTransform(alpha=0.5, sigma=30, p=0.1),  # Gardé pour la variabilité des formes
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2)  # Pour renforcer contours vivants
    ], additional_targets={"mask": "mask"})
    """

    train_transform = A.Compose([
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), rotate=(-5, 5), p=0.6),  # Plus de variation sur la taille et rotation
        A.VerticalFlip(p=0.3),  
        A.HorizontalFlip(p=0.4),  
        A.MotionBlur(blur_limit=3, p=0.07),  # Simulation de flou scanner mais limité
        #A.ElasticTransform(alpha=0.3, sigma=10, p=0.05),  # Toujours faible mais un peu plus fréquent
        A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 0.7), p=0.15),  # Renforce les contours pour éviter que le modèle prenne des petits objets flous pour vivants
        A.CoarseDropout(
            num_holes_range=(1, 2),  # Nombre de trous (min, max)
            hole_height_range=(8, 16),  # Hauteur des trous en pixels
            hole_width_range=(8, 16),  # Largeur des trous en pixels
            fill=0,  # Valeur de remplissage (ex: 0 pour noir, "random" pour bruit)
            p=0.05  # Probabilité d'application
            )        
        ], additional_targets={"mask": "mask"})
    logging.info(train_transform)
    
    # Charger une seule fois le dataset complet
    full_dataset = planktonds.PlanktonDataset(
        image_mask_dir=data_config['trainpath'],
        patch_size=data_config["patch_size"],
        mode="train",
        transform=train_transform,  # On met quand même la transformation, mais activée que pour train
        apply_transform=False  # On désactive par défaut car on va 
        #l'activer apres seulement pour le train_dataset
    )

    logging.info(f"  - I loaded {len(full_dataset)} samples")

    # Séparer les indices train/validation
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(full_dataset))


    if quick_test:  # Mode test rapide
        train_indices = indices[0:1800]
        valid_indices = indices[1800:2000]
        logging.info("Quick test mode enabled: Using a small subset of the dataset.")
    else:
        train_indices = indices[num_valid:]
        valid_indices = indices[:num_valid]

    # Créer les subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

    # Activer les transformations seulement pour train
    train_dataset.dataset.apply_transform = True  # Activer uniquement pour l'entraînement
    #valid dataset reste false.


    # Créer les dataloaders
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
    input_size = tuple(full_dataset[0][0].shape)

    logging.info(f"  - Input size is {input_size} and the number of classes is {num_classes}.")
    return train_loader, valid_loader, input_size, num_classes