# Standard imports
import argparse
import logging
import os
import pathlib
import sys

# External imports
import torch
import torchinfo.torchinfo as torchinfo
import wandb
import yaml

# Local imports
from . import data, models, utils, optim
from .utils import amp_autocast

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Initialisation de wandb si activé
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
        wandb_log = wandb.log
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # TODO : change this part to its own function
    # Charger le modèle pré-entraîné si un chemin est spécifié
    if "pretrained_model" in config and os.path.exists(config["pretrained_model"]):
        logging.info(f"Loading pretrained model from {config['pretrained_model']}")
        model.load_state_dict(
            torch.load(
                config["pretrained_model"], map_location=device, weights_only=True
            )
        )
    # else:
    # logging.warning("No pretrained model found, training from scratch.")

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"]["name"], config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config["optim"], model.parameters())

    # Build the scheduler
    logging.info("= Scheduler")
    scheduler = optim.get_scheduler(optimizer, config["scheduler"])

    # Build the logging directory
    logdir = pathlib.Path(
        utils.generate_unique_logpath(
            config["logging"]["logdir"], model_config["class"]
        )
    )
    logdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Will be logging into {logdir}")

    # Sauvegarde locale du fichier config.yaml
    config_path = logdir / "config.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Ajout du config.yaml en tant qu'Artifact dans WandB (uniquement si wandb est activé)
    if wandb_log is not None:
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(str(config_path))
        wandb.log_artifact(artifact)

    # Make a summary script of the experiment
    logging.info("= Summary")
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=next(iter(train_loader))[0].shape)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )

    # Écriture du résumé de l'expérience
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Initialisation du checkpointing
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    logging.info("= Start training")
    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        test_loss, test_f1, test_precision, test_recall = utils.test(
            model, valid_loader, loss, device
        )

        scheduler.step()  # fonctionnement du scheduler.

        # Mise à jour du checkpoint si meilleur F1-score
        updated = model_checkpoint.update(test_f1)
        logging.info(
            f"[{e}/{config['nepochs']}] Test loss : {test_loss:.4f}, Precision : {test_precision:.4f}, "
            f"Recall : {test_recall:.4f}, Test F1-score : {test_f1:.4f} "
            f"{'[>> BETTER F1 <<]' if updated else ''}"
        )

        # Log dans wandb
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log({
                "train_CE": train_loss,
                "test_CE": test_loss,
                "test_Precision": test_precision,
                "test_Recall": test_recall,
                "test_F1": test_f1,
            })


# TODO: merge test proba and test
@amp_autocast
def test(config):
    """Visualize and validate results with binary prediction."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = os.path.join(
        config["test"]["model_path"], config["test"]["model_name"]
    )

    model_config = utils.load_model_config(config["test"])

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config["data"]["testpath"], config["data"]["patch_size"], mode="test"
    )
    logging.info(f"  -  Number of test samples: {len(dataset_test)}")

    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]

    logging.info("= Model")
    model = utils.build_and_load_model(
        model_config, input_size, num_classes, model_path, device
    )

    # Seconde partie de test: Utiliser les prédiction
    logging.info("= Predict first image")
    with torch.inference_mode():
        for idx_img in range(
            dataset_test.image_patches[0][0] * dataset_test.image_patches[0][1]
        ):
            if idx_img % 400 == 0:
                logging.info(f"  - Predicting mask {idx_img}")
            image = dataset_test[idx_img].unsqueeze(0).to(device)
            dataset_test.insert(model.predict(image))

    print(dataset_test.mask_files[0][0])

    logging.info("= Compare validation image with predicted mask")
    data.show_validation_image_vs_predicted_mask(
        ds=dataset_test,  # Dataset contenant les masques prédits
        idx=0,  # Index de l'image à afficher
        validation_dataset=dataset_test,  # Dataset avec l'image de validation
        image_name="validation_vs_predicted_1.png",
    )

    # ---- Extra Evaluation Functions (using dataset_train) ----
    if config["test"].get("use_train", False):
        logging.info("= Extra evaluation using training data")
        dataset_train = data.PlanktonDataset(
            config["data"]["trainpath"], config["data"]["patch_size"]
        )
        logging.info(f"  - Number of train samples: {len(dataset_train)}")
        # For example, reconstruct image from training data or compare predicted mask vs real mask:
        logging.info("= Reconstruct image from training data")
        data.show_image_mask_from(dataset_train, 0, "image_reconstruct_1.png")
        logging.info("= Compare predicted mask to real mask")
        data.show_mask_predict_compare_to_real(
            dataset_test, 0, dataset_train, "compare_mask_1.png"
        )
    # ---------------------------------------------------------


# TODO: merge test proba and test
@amp_autocast
def test_proba(config):
    """Visualize and validate results with binary prediction."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = os.path.join(
        config["test"]["model_path"], config["test"]["model_name"]
    )

    model_config = utils.load_model_config(config["test"])

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config["data"]["testpath"], config["data"]["patch_size"], mode="test"
    )
    logging.info(f"  -  Number of test samples: {len(dataset_test)}")
    dataset_train = data.PlanktonDataset(
        config["data"]["trainpath"], config["data"]["patch_size"]
    )
    logging.info(f"  -  Number of train samples: {len(dataset_train)}")

    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]

    logging.info("= Model")
    model = utils.build_and_load_model(
        model_config, input_size, num_classes, model_path, device
    )

    # Seconde partie de test_with_proba: mask de proba prédit vs le mask binaire réel
    # or seul dataset_train à des masks binaire réels.
    logging.info("= Predict probabilities and compare to real masks")
    dataset_train_proba = data.PlanktonDataset(  # Copie dédiée aux probabilités
        config["data"]["trainpath"],
        config["data"]["patch_size"],
        mode="test",  # autorise dataset_train_proba.insert(...)
    )
    with torch.inference_mode():
        for idx_img in range(
            dataset_train.image_patches[0][0] * dataset_train.image_patches[0][1]
        ):
            if idx_img % 400 == 0:
                logging.info(f"  - Predicting probabilities for mask {idx_img}")
            image = dataset_train[idx_img][0].unsqueeze(0).to(device)

            if "segmentation_models_pytorch" in type(model).__module__:
                # Cas segmentation_models_pytorch
                logits = model(image)
                probs = torch.sigmoid(logits).half()
            else:
                probs = model.predict_probs(image).half()

            dataset_train_proba.insert(probs)

    # Visualiser le mask de proba prédit vs le mask binaire réel
    # et calcule le meilleur seuil pour obtenir le plus grand F1-score
    logging.info("= Show probabilities and compare to real masks")
    data.show_predicted_mask_proba_vs_real_mask_binary(
        dataset_train_proba, 0, dataset_train, "proba_compared_real_1.png"
    )


# TODO: merge sub and sub ensemble to a only function
"""
@amp_autocast
def sub(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    use_tta = config.get("use_tta", True)
    models_path = [
        os.path.join(model, config["test"]["model_name"])
        for model in config["test"]["model_path"]
    ]

    model_config = utils.load_model_config(config["test"])

    logging.info("= Dataset")
    dataset_test = data.PlanktonDataset(
        config["data"]["testpath"], config["data"]["patch_size"], mode="test"
    )
    logging.info(f"  -  number of sample: {len(dataset_test)}")
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]

    logging.info("= Model")
    models = [
        utils.build_and_load_model(
            model_config, input_size, num_classes, model_path, device
        )
        for model_path in models_path
    ]

    apply_sigmoid = model_config["encoder"]["model_name"] == "timm-regnety_032"

    if use_tta:
        models = [utils.apply_tta(model, use_tta) for model in models]

    logging.info("= Predict masks for all test images")
    for image_idx, (num_patches_x, num_patches_y) in enumerate(
        dataset_test.image_patches
    ):
        logging.info(f"Predicting patches for image {image_idx}")
        base_idx = sum(
            x * y for x, y in dataset_test.image_patches[:image_idx]
        )  # Precompute base index

        for idx_patch in range(num_patches_x * num_patches_y):
            global_idx = base_idx + idx_patch
            image = dataset_test[global_idx].unsqueeze(0).to(device)

            with torch.inference_mode():
                prediction = model(image).squeeze(0)

            # Apply sigmoid + threshold only for RegNetY models because it ouputs logits
            threshold = 0.5
            if apply_sigmoid:
                prediction = torch.sigmoid(prediction)
            pred_binaire = (prediction > threshold).int()
            dataset_test.insert(pred_binaire)

    logging.info("= To submit")
    dataset_test.to_submission()
"""


# TODO: merge sub and sub ensemble to a only function
@amp_autocast
def sub_ensemble(config):
    """Effectue une prédiction par ensemble de modèles."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = config.get("use_tta", True)

    # Find Models path
    models_dir_path = config["test"].get("models_dir", "~/logs/")
    models_dir_path = os.path.expanduser(models_dir_path)
    models_name = config["test"].get("models_name", "UNet_0")
    model_weight_name = config["test"].get("model_weight_name", "best_model.pt")
    model_config = config["test"].get("model_config_name", "config.yaml")

    models_dir = [
        os.path.join(models_dir_path, model_name)
        for model_name in models_name
        if os.path.isdir(os.path.join(models_dir_path, model_name))
    ]

    # Load config of each models
    models_config = []
    patch_sizes = []
    for model_dir in models_dir:
        config_path = os.path.join(model_dir, "config.yaml")
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

    # Loas Dataset
    logging.info("= Dataset")
    patch_size = patch_sizes[0]
    test_path = config["data"]["testpath"]
    dataset_test = data.PlanktonDataset(test_path, patch_size, mode="test")
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]
    logging.info(f"  -  Number of samples: {len(dataset_test)}")

    # Find Model weight files
    models_weight_path = [
        os.path.join(model_dir, model_weight_name)
        for model_dir in models_dir
        if os.path.exists(os.path.join(model_dir, model_weight_name))
    ]
    logging.info(f"Found {len(models_weight_path)} models in {models_dir}")

    if not models_weight_path:
        logging.error("No models found!")
        raise ValueError("No models fund")

    # Load Models
    logging.info("= Loading Model(s)")

    if len(models_weight_path) != len(models_config):
        raise ValueError("Mismatch size between configs and models weight")

    models = []
    for model_weight_path, model_config in zip(models_weight_path, models_config):
        model = utils.build_and_load_model(
            model_config["model"], input_size, num_classes, model_weight_path, device
        )
        if use_tta:
            model = utils.apply_tta(model, use_tta)
        models.append(model)

    if not models:
        logging.error("No valid models could be loaded!")
        raise ValueError("0 Models Loaded")
    logging.debug(f"Loaded {len(models)} models successfully!")

    # Predict Mask
    logging.info("= Predict masks")

    for image_idx, (num_patches_x, num_patches_y) in enumerate(
        dataset_test.image_patches
    ):
        logging.info(f"Predicting patches for image {image_idx}")

        for idx_patch in range(num_patches_x * num_patches_y):
            if idx_patch % 100 == 0:
                logging.info(
                    f"Predict patch {idx_patch} / {num_patches_x * num_patches_y}"
                )

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

    # **Sauvegarde de la soumission**
    logging.info("= Saving ensemble submission")
    dataset_test.to_submission()
    logging.info("Ensemble submission saved successfully!")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # check in advance if cuda is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available.")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument(
        "command",
        choices=["train", "test", "test_proba", "sub", "sub_ensemble"],
        help="Command to execute",
    )

    args = parser.parse_args()

    logging.info(f"Loading configuration from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval(f"{args.command}(config)")
