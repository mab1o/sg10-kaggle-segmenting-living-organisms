# Standard imports
import logging
import sys

# External imports
import albumentations as A
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Local imports
from . import dataloader, patch, planktonds, submission, visualization


def test_patch(config):
    """Test patch functionality."""
    logging.info("\n=== Test: Patch Extraction ===")
    dir_path = config["data"]["trainpath"]
    img_path = dir_path + "rg20091216_scan.png.ppm"
    logging.info(f"Image size: {patch.extract_ppm_size(img_path)}")
    img = patch.extract_patch_from_ppm(img_path, 4000, 4000, [1240, 1240])
    mask_path = dir_path + "rg20091216_mask.png.ppm"
    logging.info(f"Mask size: {patch.extract_ppm_size(mask_path)}")
    mask = patch.extract_patch_from_ppm(mask_path, 4000, 4000, [1240, 1240])
    visualization._show_image_mask_given(img, mask, "test_patch")
    logging.info("=== End of Test: Patch Extraction ===")


def test_plankton_dataset_train(config):
    """Test PlanktonDataset for training."""
    logging.info("\n=== Test: PlanktonDataset (Train) ===")
    dir_path = config["data"]["trainpath"]
    dataset = planktonds.PlanktonDataset(dir_path, patch_size=(10000, 10000))

    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Patch size: {dataset.patch_size}")
    logging.info(f"Image files: {len(dataset.image_files)}")
    logging.info(f"Mask files: {len(dataset.mask_files)}")

    logging.info("Printing patches for the first image...")
    num_patches_first_image = dataset.image_patches[0][0] * dataset.image_patches[0][1]
    for i in range(num_patches_first_image):
        try:
            img, mask = dataset[i]
            # visualization._show_image_mask_given(img,mask, f"test_plaktondataset_train_image_{i}")
            if i % 10 == 0:
                logging.info(f"Patch {i} processed successfully.")
        except Exception as e:
            logging.error(f"Error processing patch {i}: {e}")

    logging.info("Displaying the original image...")
    img = patch.extract_patch_from_ppm(
        dir_path + dataset.image_files[0], 0, 0, dataset.images_size[0]
    )
    mask = patch.extract_patch_from_ppm(
        dir_path + dataset.mask_files[0], 0, 0, dataset.images_size[0]
    )
    visualization._show_image_mask_given(img, mask, "test_planktondataset_train")

    logging.info("=== End of Test: PlanktonDataset (Train) ===")


def test_augmented_data(config):
    """Test PlanktonDataset for training."""
    logging.info("\n=== Test: Augmented Data ===")

    dir_path = config["data"]["trainpath"]
    logging.info("  - tranfrom = True")
    transform = A.Compose([
        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.GaussianBlur(p=0.01),
    ])

    dataset = planktonds.PlanktonDataset(
        dir_path, patch_size=(1024, 1024), transform=transform
    )
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Patch size: {dataset.patch_size}")
    logging.info(f"Image files: {len(dataset.image_files)}")
    logging.info(f"Mask files: {len(dataset.mask_files)}")

    logging.info("Displaying the original image...")
    img = patch.extract_patch_from_ppm(
        dir_path + dataset.image_files[1], 0, 0, dataset.images_size[1]
    )
    mask = patch.extract_patch_from_ppm(
        dir_path + dataset.mask_files[1], 0, 0, dataset.images_size[1]
    )
    visualization._show_image_mask_given(img, mask, "test_reconstruct_original_image_1")

    logging.info("Displaying the augmented image... (expect ugly reconstruct)")
    visualization.show_image_mask_from(dataset, 1, "test_augmented_data")
    logging.info("=== End of Test: Augmented Data ===")


def test_encoder():
    logging.info("\n=== Test: Encoder ===")
    binary_to_encode = [
        "1111",
        "111100",
        "101101001010011000111001000011011010100010001010110110100011111100001101000000",
        "111000100100000000100101111011010001010111101000001100110000001101011010101000",
    ]
    expected_results = ["l", "l", "]:Hi3JR:fSl=0", "hT0UkAGX<`=JX"]
    for binary, expected_result in zip(binary_to_encode, expected_results):
        result = submission.array_to_string(np.array(list(binary)))
        if result == expected_result:
            logging.info(f"{binary} has been correctly encode to {expected_result}.")
        else:
            logging.error(
                f"{binary} has been incorrectly encode to {result} instead of {expected_result}."
            )
    logging.info("=== End of Test: Encoder ===")


def test_reconstruct_image(config, original=False):
    """Test image reconstruction."""
    logging.info("\n=== Test: Image Reconstruction ===")

    dir_path = config["data"]["trainpath"]
    patch_size = (10000, 10000)
    dataset = planktonds.PlanktonDataset(dir_path, patch_size)

    if original:
        logging.info("Displaying the original image...")
        img = patch.extract_patch_from_ppm(
            dir_path + dataset.image_files[1], 0, 0, dataset.images_size[1]
        )
        mask = patch.extract_patch_from_ppm(
            dir_path + dataset.mask_files[1], 0, 0, dataset.images_size[1]
        )
        visualization._show_image_mask_given(
            img, mask, "test_reconstruct_original_image_1"
        )

    logging.info("Displaying the reconstructed image...")
    visualization.show_image_mask_from(dataset, 1, "test_reconstruct_image")

    logging.info("=== End of Test: Image Reconstruction ===")


def test_generate_csv_file(config):
    logging.info("\n=== Test: Generation fichier CSV ===")

    logging.info("Charge Data")
    dir_path = config["data"]["trainpath"]
    patch_size = (20000, 2)
    ds = planktonds.PlanktonDataset(dir_path, patch_size)

    logging.info("Make csv file")
    _, mask = ds[10000]
    _, mask1 = ds[11000]
    ordered_list = ds.mask_files[:2]
    submission.generate_submission_file(
        [mask, mask1], ordered_list, "test_csv_file.csv"
    )

    logging.info("=== End of Test: Generation fichier CSV ===")


def test_plankton_dataset_inference(config):
    logging.info("\n=== Test: PlanktonDataset (Test) ===")

    logging.info("Charge Data train")
    logging.info("  - tranfrom = True")
    transform = A.Compose([
        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.GaussianBlur(p=0.01),
    ])
    dir_path = config["data"]["trainpath"]
    patch_size = (1000, 1000)
    ds_train = planktonds.PlanktonDataset(dir_path, patch_size, transform=transform)

    logging.info("Charge Data test")
    ds_test = planktonds.PlanktonDataset(dir_path, patch_size, mode="test")

    num_patches_first_image = (
        ds_train.image_patches[0][0] * ds_train.image_patches[0][1]
    )
    for i in range(num_patches_first_image):
        if i % 10 == 0:
            logging.info(f"Processing index {i}...")
        try:
            ds_test.insert(ds_train[i][1])
        except Exception as e:
            logging.info(f"Error at index {i}: {e}")

    logging.info("Impression de l'image reconstruite")
    visualization.show_mask_predict_compare_to_real(
        ds_test, 0, ds_train, "test_planktondataset_test"
    )

    logging.info("=== End of Test: PlanktonDataset (Test) ===")


def test_dataloader(config):
    """Test data loader functionality."""
    logging.info("\n=== Test: DataLoader ===")

    train_loader, valid_loader, input_size, num_classes = dataloader.get_dataloaders(
        config["data"], False
    )
    logging.info(f"Train loader size: {len(train_loader)}")
    logging.info(f"Validation loader size: {len(valid_loader)}")
    logging.info(f"Input size: {input_size}")
    logging.info(f"Number of classes: {num_classes}")

    logging.info("=== End of Test: DataLoader ===")


def test_size_plankton(config):
    """Test the size of PlanktonDataset samples."""
    logging.info("\n=== Test: PlanktonDataset Size ===")

    dir_path = config["data"]["trainpath"]
    patch_size = (10000, 10000)
    dataset = planktonds.PlanktonDataset(dir_path, patch_size)

    img, mask = dataset[0]
    logging.info(f"Image shape: {img.shape}")
    logging.info(f"Mask shape: {mask.shape}")

    assert len(img.shape) == 3, "Image should have 3 dimensions (C, H, W)."
    assert len(mask.shape) == 2, "Mask should have 2 dimensions (H, W)."

    logging.info("=== End of Test: PlanktonDataset Size ===")


def test_train_transform():
    logging.info("\n=== Test: Transform train ===")
    train_transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-10, 10),
                p=0.5,
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

    dataset = planktonds.PlanktonDataset(
        "/mounts/Datasets3/2024-2025-ChallengePlankton/train/",
        patch_size=(4096, 4096),
        mode="train",
        transform=train_transform,
        apply_transform=True,
    )

    # Afficher un exemple
    image, mask = dataset[0]  # Premier exemple
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image.squeeze(), cmap="gray")
    axs[0].set_title("Image transformée")
    axs[1].imshow(mask.squeeze(), cmap="gray")
    axs[1].set_title("Masque transformé")
    plt.savefig("test_output.png")

    logging.info("=== End of Test: Transform train ===")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage : {sys.argv[0]} config.yaml ")
        sys.exit(-1)

    logging.info(f"Loading {sys.argv[1]}")
    config = yaml.safe_load(open(sys.argv[1]))

    # test_patch(config)
    # test_PlanktonDataset_train(config)
    # test_reconstruct_image(config)
    # test_dataloader(config)
    # test_size_plankton(config)
    # test_encoder()
    # test_generate_csv_file(config)
    # test_PlanktonDataset_test(config)
    # test_augmented_data(config)
    test_train_transform()
