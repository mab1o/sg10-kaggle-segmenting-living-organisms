import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(transform_type="light"):
    """
    Retourne un ensemble de transformations Albumentations en fonction du type demandé.

    Args:
        transform_type (str): Type de transformation. Options :
            - "none"    : Aucune transformation
            - "light"   : Transformations légères    0.897
            - "medium"  : Transformations modérées   0.903
            - "high"    : Transformations agressives
    Returns:
        albumentations.Compose : Transformation sélectionnée.
    """

    if transform_type == "none":
        return A.Compose([])

    elif transform_type == "light":
        return A.Compose([
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.GaussianBlur(p=0.01)
        ])

    elif transform_type == "medium":
        return A.Compose([
            A.Affine(
                scale=(0.95, 1.05), 
                translate_percent=(0.02, 0.02), 
                rotate=(-5, 5), 
                p=0.6
            ),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.4),
            A.MotionBlur(blur_limit=3, p=0.07),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 0.7), p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 2), 
                hole_height_range=(8, 16), 
                hole_width_range=(8, 16),
                fill=0, p=0.05
            )
        ])

    elif transform_type == "high":
        return A.Compose([
            A.Affine(
                scale=(0.90, 1.10),  # Augmente encore la variabilité de la taille
                translate_percent=(0.03, 0.03),  # Laisse plus de liberté aux translations
                rotate=(-10, 10),  # Rotation un peu plus forte
                shear=(-5, 5),  # Ajout d'un léger effet de cisaillement
                p=0.7
            ),
            A.VerticalFlip(p=0.35),
            A.HorizontalFlip(p=0.45),
            A.MotionBlur(blur_limit=5, p=0.1),  # Augmente le flou
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.4, 0.8), p=0.2),  # Légèrement plus agressif
            A.CoarseDropout(
                num_holes_range=(1, 3),  # Ajoute une occlusion un peu plus forte
                hole_height_range=(8, 20),
                hole_width_range=(8, 20),
                fill=0, p=0.07
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1)  # Ajoute une légère déformation globale
        ])
    # Default to light if incorrect input
    return get_transforms("light")
