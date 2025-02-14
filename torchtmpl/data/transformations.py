import albumentations as A


def get_transforms(transform_type="light"):
    """Retourne un ensemble de transformations Albumentations en fonction du type demandé.

    Args: transform_type (str): Type de transformation. Options :
            - "none"    : Aucune transformation
            - "light"   : Transformations légères    0.897
            - "medium"  : Transformations modérées   0.903
            - "high"    : Transformations agressives

    Returns: albumentations.Compose : Transformation sélectionnée.
    """
    base_transforms = {
        "none": [],
        "light": [
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.GaussianBlur(p=0.01),
        ],
        "medium": [
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
        ],
        "high": [
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(0.03, 0.03),
                rotate=(-10, 10),
                shear=(-5, 5),
                p=0.7,
            ),
            A.VerticalFlip(p=0.35),
            A.HorizontalFlip(p=0.45),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.4, 0.8), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 20),
                hole_width_range=(8, 20),
                fill=0,
                p=0.07,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1),
        ],
    }

    extra_transforms = {
        "medium-light": [
            A.Affine(
                scale=(0.93, 1.07),
                translate_percent=(0.03, 0.03),
                rotate=(-7, 7),
                p=0.65,
            ),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.4, 0.6), sigma=0.7, p=0.12),
            A.CoarseDropout(
                num_holes_range=(2, 4),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0,
                p=0.08,
            ),
        ],
        "medium-best": [
            A.Affine(
                scale=(0.94, 1.06),
                translate_percent=(0.025, 0.025),
                rotate=(-6, 6),
                p=0.65,
            ),
            A.VerticalFlip(p=0.35),
            A.HorizontalFlip(p=0.45),
            A.MotionBlur(blur_limit=3, p=0.06),
            A.Sharpen(alpha=(0.18, 0.38), lightness=(0.5, 0.7), p=0.13),
            A.CoarseDropout(
                num_holes_range=(2, 3),
                hole_height_range=(10, 18),
                hole_width_range=(10, 18),
                fill=0,
                p=0.06,
            ),
        ],
        "medium-heavy": [
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-5, 5),
                p=0.6,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.1),
            A.RandomBrightnessContrast(p=0.2),
        ],
    }

    test_transforms = {
        "test_conservative": [
            A.GaussianBlur(blur_limit=(1, 3), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            A.CoarseDropout(
                num_holes_range=(2, 2),
                hole_height_range=(10, 15),
                hole_width_range=(10, 15),
                fill=0,
                p=0.08,
            ),
        ],
        "test_moderate": [
            A.ElasticTransform(alpha=1, sigma=20, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.25
            ),
            A.CoarseDropout(
                num_holes_range=(3, 3),
                hole_height_range=(15, 20),
                hole_width_range=(15, 20),
                fill=0,
                p=0.1,
            ),
        ],
        "test_extreme": [
            A.ElasticTransform(alpha=2, sigma=25, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.CoarseDropout(
                num_holes_range=(4, 4),
                hole_height_range=(20, 25),
                hole_width_range=(20, 25),
                fill=0,
                p=0.15,
            ),
        ],
    }

    # Combine base and additional transforms
    if transform_type == "none":
        return A.Compose([])

    if transform_type in base_transforms:
        return A.Compose(base_transforms[transform_type])

    if transform_type in extra_transforms:
        return A.Compose(base_transforms["medium"] + extra_transforms[transform_type])

    if transform_type in test_transforms:
        return A.Compose(base_transforms["medium"] + test_transforms[transform_type])

    raise ValueError(f"{transform_type} is an unknown transformation")
