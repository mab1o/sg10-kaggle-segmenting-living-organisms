import albumentations as A


def get_transforms(transform_type="light"):  # noqa: C901
    """Retourne un ensemble de transformations Albumentations en fonction du type demandé.

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
            A.GaussianBlur(p=0.01),
        ])

    elif transform_type == "medium-light":
        return A.Compose([
            A.Affine(
                scale=(0.93, 1.07),
                translate_percent=(0.03, 0.03),
                rotate=(-7, 7),
                p=0.65,
            ),  # More variation
            A.VerticalFlip(p=0.4),
            A.HorizontalFlip(p=0.4),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Sharpen(
                alpha=(0.1, 0.3), lightness=(0.4, 0.6), sigma=0.7, p=0.12
            ),  # Reduced sharpening
            A.CoarseDropout(
                num_holes_range=(2, 4),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0,
                p=0.08,
            ),  # More aggressive dropout
        ])

    elif transform_type == "medium":
        return A.Compose([
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-5, 5),
                p=0.6,
            ),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.4),
            A.MotionBlur(blur_limit=3, p=0.07),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 0.7), p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.05,
            ),
        ])

    elif transform_type == "medium-best":
        return A.Compose([
            A.Affine(
                scale=(0.94, 1.06),  # Légère augmentation de la variation
                translate_percent=(0.025, 0.025),  # Un peu plus que Medium
                rotate=(-6, 6),  # Rotation légèrement plus forte
                p=0.65,
            ),
            A.VerticalFlip(p=0.35),  # Légère augmentation
            A.HorizontalFlip(p=0.45),  # Idem, plus équilibré
            A.MotionBlur(
                blur_limit=3, p=0.06
            ),  # Diminué légèrement pour éviter de flouter trop
            A.Sharpen(
                alpha=(0.18, 0.38), lightness=(0.5, 0.7), p=0.13
            ),  # Moins fort que High, mais plus que Medium
            A.CoarseDropout(
                num_holes_range=(2, 3),  # Augmenté pour plus de robustesse
                hole_height_range=(10, 18),
                hole_width_range=(10, 18),
                fill=0,
                p=0.06,
            ),
        ])

    elif transform_type == "medium-heavy":
        return A.Compose([
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-5, 5),
                p=0.6,
            ),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.4),
            A.MotionBlur(blur_limit=3, p=0.07),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 0.7), p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.05,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1),  # ↑ Ajouté
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),  # ↑ Ajouté
            A.RandomBrightnessContrast(p=0.2),  # ↑ Ajouté
        ])

    elif transform_type == "high":
        return A.Compose([
            A.Affine(
                scale=(0.90, 1.10),  # Augmente encore la variabilité de la taille
                translate_percent=(
                    0.03,
                    0.03,
                ),  # Laisse plus de liberté aux translations
                rotate=(-10, 10),  # Rotation un peu plus forte
                shear=(-5, 5),  # Ajout d'un léger effet de cisaillement
                p=0.7,
            ),
            A.VerticalFlip(p=0.35),
            A.HorizontalFlip(p=0.45),
            A.MotionBlur(blur_limit=5, p=0.1),  # Augmente le flou
            A.Sharpen(
                alpha=(0.2, 0.5), lightness=(0.4, 0.8), p=0.2
            ),  # Légèrement plus agressif
            A.CoarseDropout(
                num_holes_range=(1, 3),  # Ajoute une occlusion un peu plus forte
                hole_height_range=(8, 20),
                hole_width_range=(8, 20),
                fill=0,
                p=0.07,
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.03, p=0.1
            ),  # Ajoute une légère déformation globale
        ])

    elif transform_type == "test_conservative":
        return A.Compose([
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-5, 5),
                p=0.65,
            ),
            A.VerticalFlip(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            A.CoarseDropout(
                max_holes=2, max_height=15, max_width=15, fill_value=0, p=0.08
            ),
        ])

    elif transform_type == "test_moderate":
        return A.Compose([
            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(0.03, 0.03),
                rotate=(-8, 8),
                p=0.7,
            ),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.25
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),
            A.CoarseDropout(
                max_holes=3, max_height=20, max_width=20, fill_value=0, p=0.1
            ),
        ])

    elif transform_type == "test_extreme":
        return A.Compose([
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(0.04, 0.04),
                rotate=(-12, 12),
                p=0.75,
            ),
            A.VerticalFlip(p=0.55),
            A.HorizontalFlip(p=0.55),
            A.ElasticTransform(alpha=2, sigma=25, alpha_affine=15, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
            A.CoarseDropout(
                max_holes=4, max_height=25, max_width=25, fill_value=0, p=0.15
            ),
        ])

    # Default to light if incorrect input
    print("No transformation selected")
    print("Defaulting to light transformations")

    return get_transforms("light")
