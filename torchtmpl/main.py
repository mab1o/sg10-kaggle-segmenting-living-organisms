# Standard imports
import argparse
import logging
import os
import sys

# External imports
import torch
import wandb
import yaml

# Local imports
from . import data, models, optim, utils
from .utils import amp_autocast

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define level of log
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(config):
    """Train a model based on the provided configuration.

    This function:
    - Builds the dataloaders for training and validation datasets.
    - Configures the loss function, optimizer, and learning rate scheduler.
    - Creates directories to save logs and model weights.
    - Saves model checkpoints based on performance metrics.

    Arguments:
    - config: A dictionary containing configuration settings for the model, optimizer, loss function, etc.

    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Check if first train or continue train
    first_train, config, logdir = utils.update_config_train(config)

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

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"]["name"], config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config["optim"], model.parameters())

    # Build the scheduler
    logging.info("= Scheduler")
    scheduler = optim.get_scheduler(optimizer, config["scheduler"])

    # Create dir to save all informations and the weight of the model
    if first_train:
        logdir = utils.create_doc_save_model(
            config, wandb_log, train_loader, valid_loader, model_config, model, loss
        )
        utils.save_datasets_indice(
            train_loader, valid_loader, str(logdir / "dataset.pt")
        )

    # Initialisation of checkpoint if needed
    start_epoch, best_loss = utils.load_checkpoint(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        str(logdir / "checkpoint.pt"),
        str(logdir / "dataset.pt"),
        device,
    )

    # Initialisation du checkpointing
    model_checkpoint = utils.ModelCheckpoint(
        model,
        optimizer,
        scheduler,
        str(logdir / "best_model.pt"),
        str(logdir / "checkpoint.pt"),
        min_is_best=False,
        best_score=best_loss,
    )

    logging.info("= Start training")
    for e in range(start_epoch, config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(
            model, train_loader, loss, optimizer, scheduler, device
        )
        test_loss, test_f1, test_precision, test_recall = utils.test(
            model, valid_loader, loss, device
        )

        # Mise à jour du checkpoint si meilleur F1-score
        updated = model_checkpoint.update(test_f1, e)
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


@amp_autocast
def test(config):
    """Evaluates the trained model on a test dataset and visualizes the results.

    This function:
    - Loads the test and training datasets (if required for additional evaluation).
    - Loads the pre-trained model using the provided weights.
    - Makes predictions on the test dataset.
    - Displays comparisons between the predicted masks and the real masks.
    - Optionally calculates and visualizes probabilities, as well as performs extra evaluations on the training data.

    Arguments:
    - config: A dictionary containing configuration settings for the model, dataset, and evaluation process.

    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models_dir_path = config["test"].get("models_dir", "~/logs/")
    models_dir_path = os.path.expanduser(models_dir_path)
    models_name = config["test"].get("models_name", ["UNet_0"])
    model_weight_name = config["test"].get("model_weight_name", "best_model.pt")
    model_config_name = config["test"].get("model_config_name", "config.yaml")

    model_dir = os.path.join(models_dir_path, models_name[0])
    model_path = os.path.join(model_dir, model_weight_name)
    model_config = utils.load_model_config(model_dir, model_config_name)

    # Load datasets
    logging.info("= Dataset")
    dataset_test, dataset_train = utils.load_dataset(config, model_config)
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]

    # Load Model
    logging.info("= Model")
    model = utils.build_and_load_model(
        model_config["model"], input_size, num_classes, model_path, device
    )

    # Predict First Image
    logging.info("= Predict first image")
    utils.predict_and_insert(dataset_test, model, device)

    # Show image vs maxk predict
    logging.info("= Compare validation image with predicted mask")
    data.show_validation_image_vs_predicted_mask(
        ds=dataset_test,
        idx=0,
        validation_dataset=dataset_test,
        image_name="validation_vs_predicted_1.png",
    )

    # Show probabilities and compare to real masks
    if dataset_train is not None:
        utils.calc_and_show_proba(config, device, model_config, dataset_train, model)

    # Extra Evaluation Functions (using training data)
    if config["test"].get("use_train", False):
        utils.more_eval(dataset_test, dataset_train)


@amp_autocast
def sub(config):
    """Perform model inference for a set of models and generates a submission file.

    This function:
    - Loads multiple models and their respective weights.
    - Makes predictions for each model on the test dataset.
    - Combines the results and generates a submission file.

    Arguments:
    - config: A dictionary containing configuration settings for the models, dataset, and submission process.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = config.get("use_tta", True)
    models_dir_path = config["test"].get("models_dir", "~/logs/")
    models_dir_path = os.path.expanduser(models_dir_path)
    models_name = config["test"].get("models_name", "UNet_0")
    model_weight_name = config["test"].get("model_weight_name", "best_model.pt")
    model_config_name = config["test"].get("model_config_name", "config.yaml")

    # Find Models Path
    models_dir = [
        os.path.join(models_dir_path, model_name)
        for model_name in models_name
        if os.path.isdir(os.path.join(models_dir_path, model_name))
    ]

    # Load Configs
    models_config, patch_sizes = utils.load_configs(models_dir, model_config_name)

    # Load Dataset
    logging.info("= Dataset")
    test_path = config["data"]["testpath"]
    dataset_test = data.PlanktonDataset(test_path, patch_sizes[0], mode="test")
    input_size = tuple(dataset_test[0].shape)
    num_classes = input_size[0]
    logging.info(f"  -  Number of samples: {len(dataset_test)}")

    # Find Model Weight Files
    models_weight_path = utils.find_models_weight_path(model_weight_name, models_dir)

    # Load Models
    logging.info("= Loading Model(s)")
    models = utils.load_models(
        device, use_tta, models_config, input_size, num_classes, models_weight_path
    )

    # Predict Masks
    logging.info("= Predict masks")
    utils.predict_masks(device, dataset_test, models)

    # Create submission file
    logging.info("= Creating submission")
    dataset_test.to_submission()
    logging.info("Submission saved successfully!")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # Check in advance if cuda is available
    if torch.cuda.is_available():
        logging.info(f"CUDA is available! Device name: {torch.cuda.get_device_name(0)}")
    else:
        logging.error("CUDA is NOT available.")

    # Lauch command
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument(
        "command",
        choices=["train", "test", "sub"],
        help="Command to execute",
    )

    args = parser.parse_args()

    logging.info(f"Loading configuration from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval(f"{args.command}(config)")
