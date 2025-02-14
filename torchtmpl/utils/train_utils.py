import datetime
import functools
import os
import logging
import pathlib
import sys

# External imports
import segmentation_models_pytorch as smp
import torch
import torch.nn
import tqdm
from torch.amp import GradScaler, autocast
import torchinfo.torchinfo as torchinfo
import wandb
import yaml


def amp_autocast(func):
    """Enable automatic mixed precision (AMP) for the wrapped function (decorator).

    Args:
        func (function): The function to wrap.

    Returns:
        wrapper (function): The wrapped function with AMP enabled.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.amp.autocast(device_type="cuda"):
            return func(*args, **kwargs)

    return wrapper


def generate_unique_logpath(logdir, raw_run_name):
    """Generate a unique log path for saving model logs.

    Args:
        logdir (str): The relative path for logs from the user's home directory.
        raw_run_name (str): The base name for the run (typically the experiment name).

    Returns:
        log_path (str): A unique path to store the logs.

    """
    # Ensure logdir is stored in HOME while keeping it relative in YAML
    home_logdir = os.path.join(
        os.path.expanduser("~"), logdir
    )  # Expands ~/logs → /home/username/logs

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # AAAAMMJJ-HHMMSS
    i = 0

    while True:
        run_name = f"{raw_run_name}_{i}_{timestamp}"
        log_path = os.path.join(home_logdir, run_name)

        if not os.path.isdir(log_path):
            os.makedirs(log_path, exist_ok=True)
            return log_path
        i += 1


class ModelCheckpoint:
    """Callback class to save the model whenever a better score is achieved.

    Args:
        model (torch.nn.Module): The model to monitor.
        savepath (str): The file path to save the model.
        min_is_best (bool): If True, the model with the lowest score is considered the best.
                            If False, the model with the highest score is considered the best.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath: str,
        min_is_best: bool = True,
    ) -> None:
        """Initialize a modelCheckpint instance.

        Args:
            model (torch.nn.Module): The model to monitor.
            savepath (str): The file path where to save the model
            min_is_best (bool, optional): Boolean to keep the lowest score. Defaults to True.

        """
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, scheduler, device, dynamic_display=True):
    """Train the model for one epoch using the provided data loader.

    Args:
        model (torch.nn.Module): The model to train.
        loader (DataLoader): The data loader to provide training data.
        f_loss (function): The loss function to compute the loss.
        optimizer (Optimizer): The optimizer to update the model's parameters.
        scheduler (Scheduler): The scheduler to adjust the learning rate.
        device (torch.device): The device (CPU or GPU) for model training.
        dynamic_display (bool): Whether to display a dynamic progress bar (default: True).

    Returns:
        avg_loss (float): The average loss over the entire training epoch.

    """
    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    scaler = GradScaler("cuda")  # Initialiser le scaler pour la précision mixte

    for _i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Toujours remettre les gradients à zéro avant le passage avant

        # Forward pass avec précision mixte
        with autocast("cuda"):
            outputs = model(inputs).squeeze(1)
            loss = f_loss(outputs, targets)

        # Backward pass et optimisation
        scaler.scale(loss).backward()

        # Clipping des gradients pour éviter les explosions de gradients
        # scaler.unscale_(optimizer)  # Nécessaire avant clip_grad_norm_ avec AMP
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss / num_samples:.4f}")

    # Scheduler update
    scheduler.step()
    return total_loss / num_samples


@amp_autocast
def test(model, loader, f_loss, device):
    """Test the model over the loader using specified metrics (loss, precision, recall, F1-score).

    Args:
        model (torch.nn.Module): The model to test.
        loader (DataLoader): The data loader for the test data.
        f_loss (function): The loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) for evaluation.

    Returns:
        avg_loss (float): The average loss over the entire test set.
        avg_f1 (float): The average F1-score over the entire test set.
        avg_precision (float): The average precision over the entire test set.
        avg_recall (float): The average recall over the entire test set.

    """
    model.eval()
    total_loss, num_samples = 0, 0

    # Initialize true positive, false positive, false negative, true negative
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze(1)
            loss = f_loss(outputs, targets)

            # Compute loss sum
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

            # Convert logits to binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).int()

            targets = targets.int()  # Ajoute cette ligne

            # Get stats for precision, recall, F1 computation
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                preds, targets, mode="binary", threshold=0.5
            )

            # Aggregate stats
            tp += batch_tp.sum()
            fp += batch_fp.sum()
            fn += batch_fn.sum()
            tn += batch_tn.sum()

    # Compute final metrics
    avg_loss = total_loss / num_samples
    avg_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    avg_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
    avg_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

    return avg_loss, avg_f1, avg_precision, avg_recall


def create_doc_save_model(
    config, wandb_log, train_loader, valid_loader, model_config, model, loss
):
    """Create and saves a summary document for the training experiment and logs it to the specified location.

    Args:
        config (dict): The configuration dictionary.
        wandb_log (object): If not None, logs to Weights & Biases.
        train_loader (DataLoader): The training data loader.
        valid_loader (DataLoader): The validation data loader.
        model_config (dict): The configuration dictionary for the model.
        model (torch.nn.Module): The trained model.
        loss (str): A string description of the loss function used.

    Returns:
        logdir (Path): The directory where the logs and model summary are saved.

    """
    # Build the logging directory
    logdir = pathlib.Path(
        generate_unique_logpath(config["logging"]["logdir"], model_config["class"])
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
    return logdir
