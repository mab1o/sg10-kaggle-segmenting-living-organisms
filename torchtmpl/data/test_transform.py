# External imports
import matplotlib.pyplot as plt
import albumentations as A

# Local imports
from torchtmpl.data.planktonds import PlanktonDataset

train_transform = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(1, 3), p=0.2),  # Replaced MotionBlur
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50, p=0.2
        ),  # Added for shape deformation
    ],
    additional_targets={"mask": "mask"},
)


dataset = PlanktonDataset(
    "/mounts/Datasets3/2024-2025-ChallengePlankton/train/",
    patch_size=(4096, 4096),
    mode="train",
    transform=train_transform,  # Assure-toi que transform est bien accepté
    apply_transform=True,  # Important pour appliquer les transformations
)
# Afficher un exemple
image, mask = dataset[0]  # Premier exemple
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image.squeeze(), cmap="gray")
axs[0].set_title("Image transformée")
axs[1].imshow(mask.squeeze(), cmap="gray")
axs[1].set_title("Masque transformé")
plt.savefig("test_output.png")
